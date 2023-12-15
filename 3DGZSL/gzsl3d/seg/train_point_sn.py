import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax
from torch_cluster import fps
from tqdm import tqdm

# Imports for Backbone
sys.path.append('..')
from fkaconv.lightconvpoint.utils import transformations as lcp_transfo
from fkaconv.lightconvpoint.networks.fkaconv import FKAConv as Network
from fkaconv.examples.scannet.train import get_data

# Imports from 3DGZSL modules
from dataloaders import make_data_loader
from utils.loss import SegmentationLosses, InfoNCELosses # , TripletLoss
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from class_names import CLASS_NAMES_SN
from trainer_class import Trainer_default, main

def rel_distance(pc1, pc2, pc_distance=False):
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).repeat(1, 1, M, 1)
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1, N, 1, 1)
    pc_diff = pc1_expand_tile - pc2_expand_tile

    pc_dist = torch.sum(pc_diff ** 2, dim=-1)  # (B,N,M)
    dist1, idx1 = torch.min(pc_dist, dim=2)  # (B,N)
    dist2, idx2 = torch.min(pc_dist, dim=1)  # (B,M)
    if pc_distance:
        return pc_dist
    else:
        return dist1, idx1, dist2, idx2

class Trainer(Trainer_default):

    def __init__(self, args):
        super().__init__(args)
        # Define Dataloader
        self.N_CLASSES = 21
        training_transformations_data = [lcp_transfo.RandomSubSample(args.config["dataset_num_points"])]
        validation_transformations_data = [lcp_transfo.RandomSubSample(args.config["dataset_num_points"])]
        training_transformations_points = [lcp_transfo.RandomRotation(rotation_axis=2),]
        self.class_names = CLASS_NAMES_SN

        def network_function():
            return Network(3, self.N_CLASSES, segmentation=True)

        (self.train_loader, self.val_loader, _, self.nclass,) = make_data_loader(
            args, data_dir=args.rootdir, network_function=network_function,
            training_transformations_points=training_transformations_points,
            training_transformations_data=training_transformations_data,
            validation_transformations_data=validation_transformations_data)
        device = torch.device("cuda")
        net = network_function()
        net.to(device)
        model = net
        #Load the ConvBasic version
        if args.test:
            chkp_path = os.path.join(args.savedir)
            print("Load the complete trained model from {}".format(chkp_path))
            checkpoint = torch.load(chkp_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            print("Load the pre-trained network from {}".\
                format(os.path.join(args.config_savedir, "checkpoint.pth")))
            model.load_state_dict(torch.load(os.path.join(args.config_savedir,\
                 "checkpoint.pth"))['state_dict'], strict=False)

        #Freeze all layers besides the linear layer that gets finetuned
        model.freeze_backbone_fcout()
        model.freeze_bn()

        train_params = [{"params": model.get_1x_lr_params(), "lr": args.lr},\
            {"params": model.get_10x_lr_params(), "lr": args.lr * 10}]

        # Define Optimizer adn generator optimizer
        optimizer = torch.optim.Adam(train_params, lr=args.config["lr_start"])

        # Define Generator
        embed_feature_size = 0
        self.set_generator(embed_feature_size=embed_feature_size)

        class_weight = torch.ones(self.nclass)
        class_weight[args.unseen_classes_idx_metric] = args.unseen_weight

        if args.cuda:
            class_weight = class_weight.cuda()
        self.criterion = SegmentationLosses(weight=class_weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.cst_criterion = InfoNCELosses(temperature=args.temperature, negative_mode='unpaired')
        self.model, self.optimizer = model, optimizer

        # # Define Evaluator
        self.evaluator = Evaluator(self.nclass, args.seen_classes_idx_metric, args.unseen_classes_idx_metric)

        # # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()
            self.generator = self.generator.cuda()

        # # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            if args.cuda:
                self.model.load_state_dict(checkpoint["state_dict"])

        # # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

        self.s2vmapper = nn.Linear(args.embed_dim, 64).cuda()

    def getMutiplePrototypes(self, feat, n_protos):
        """
        Extract multiple prototypes by points separation and assembly
        Args:
            feat: input point features, shape:(n_points, feat_dim)
        Return:
            prototypes: output prototypes, shape: (n_prototypes, feat_dim)
        """
        # sample n_protos seeds as initial centers with Farthest Point Sampling (FPS)
        n, feat_dim = feat.shape[0], feat.shape[1]
        assert n > 0
        ratio = n_protos / n
        if ratio < 1:
            fps_index = fps(feat, None, ratio=ratio, random_start=False).unique()
            num_prototypes = len(fps_index)
            farthest_seeds = feat[fps_index]

            # compute the point-to-seed distance
            distances = F.pairwise_distance(feat[..., None], farthest_seeds.transpose(0, 1)[None, ...],
                                            p=2)  # (n_points, n_prototypes)

            # hard assignment for each point
            assignments = torch.argmin(distances, dim=1)  # (n_points,)

            # aggregating each cluster to form prototype
            prototypes = torch.zeros((num_prototypes, feat_dim)).cuda()
            for i in range(num_prototypes):
                selected = torch.nonzero(assignments == i).squeeze(1)
                selected = feat[selected, :]
                prototypes[i] = selected.mean(0)
            return prototypes
        else:
            return feat

    def training(self, epoch, args):
        train_loss = 0.0
        self.model.module.train()
        self.model.module.freeze_backbone_fcout()
        self.model.module.freeze_bn()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        i = None
        last_i = -1

        for i, sample in enumerate(tbar):

            features, pts, target, net_ids, net_pts, idx_batch, embedding =\
                get_data(sample, attributes=True, device="cuda")

            target = target[:, :, None]

            if self.args.cuda:
                pts, features, target, embedding = (
                    pts.cuda(),
                    features.cuda(),
                    target.cuda(),
                    embedding.cuda()
                )
            self.scheduler(self.optimizer, i, epoch, self.best_pred)

            with torch.no_grad():
                real_features = self.model.module.backbone(features, pts, support_points=net_pts, indices=net_ids)
            real_features = real_features.permute(0, 2, 1)

            # ===================Generator training=====================
            fake_features = torch.zeros(real_features.shape)
            if args.cuda:
                fake_features = fake_features.cuda()
            generator_loss_batch = 0.0

            for (count_sample_i, (real_features_i, pts_i, target_i, embedding_i)) in enumerate(zip(real_features, pts, target, embedding)):
                generator_loss_sample = 0.0

                target_i = target_i.view(-1)
                fake_features_i = torch.zeros(real_features_i.shape)
                if args.cuda:
                    fake_features_i = fake_features_i.cuda()

                unique_class = torch.unique(target_i)

                # If PC has a point belonging to unseen class pixel, no training for generator and generated features for the whole PC
                has_unseen_class = False
                for u_class in unique_class:
                    if u_class in args.unseen_classes_idx_metric:
                        has_unseen_class = True

                if i != last_i:
                    cur_visual_clster = {}
                    cur_sematic_clster = {}
                    for unseen in args.unseen_classes_idx_metric[1:]:
                        cur_visual_clster[int(unseen)] = torch.zeros([1, 64]).cuda()
                        cur_sematic_clster[int(unseen)] = torch.zeros([1, 600]).cuda()
                    last_i = i
                else:
                    last_visual_clster = cur_visual_clster.copy()
                    last_sematic_clster = cur_sematic_clster.copy()

                for idx_in in unique_class:
                    if idx_in != 255:
                        self.optimizer_generator.zero_grad()

                        idx_class = target_i == idx_in
                        real_features_class = real_features_i[idx_class]
                        embedding_class = embedding_i[idx_class]
                        neg_point_features = real_features_i[target_i != idx_in]
                        s_protos = embedding_class[0]
                        s_cluster= embedding_class[0].unsqueeze(0)

                        # Noise generation
                        z = torch.rand((embedding_class.shape[0], args.noise_dim))
                        if args.cuda:
                            z = z.cuda()

                        # Generation of the features
                        if args.generator_model == "gmmn":
                            if not has_unseen_class:
                                select = random.sample(range(embedding_class.shape[0]),
                                                       int(args.mask_ratio * embedding_class.shape[0]))
                                mask = torch.ones([embedding_class.shape[0], 1])
                                mask[select] = 0.0
                                embedding_class = embedding_class * mask.cuda()

                            fake_features_class = self.generator(embedding_class, z.float())
                            fake_feature_train = fake_features_class
                        else:
                            print("Incorrect generator model selection")
                            raise NotImplementedError

                        s_protos = self.s2vmapper(s_protos.unsqueeze(0))

                        if (idx_in in args.seen_classes_idx_metric and not has_unseen_class):
                            # Avoid CUDA out of memory
                            random_idx = torch.randint(low=0, high=fake_feature_train.shape[0], size=(args.batch_size_generator,))

                            n_protos = max(int(real_features_class.shape[0] * args.proto_ratio), 1)
                            visual_protos = self.getMutiplePrototypes(real_features_class, n_protos)  # [n_protos, 64]

                            s2v_loss = 0
                            for v in range(visual_protos.shape[0]):
                                s2v_loss += (1 - F.cosine_similarity(visual_protos[v, :].unsqueeze(0), s_protos, dim=1)) / 2
                            s2v_loss = s2v_loss / (visual_protos.shape[0])

                            real_features_pos = real_features_class[random_idx]
                            fake_features_cur = fake_feature_train[random_idx]

                            _, D = real_features_pos.shape
                            neg_point_features = neg_point_features.reshape(-1, D)

                            contrast_loss = self.cst_criterion(fake_features_cur, real_features_pos, neg_point_features)
                            contrast_loss.requires_grad_(requires_grad=True)

                            visual_clster_list = []
                            sematic_clster_list = []
                            if i==0 and count_sample_i == 0:
                                g_loss = self.criterion_generator(
                                    fake_features_cur + s_protos,
                                    real_features_pos, )
                                total_loss = g_loss + contrast_loss + s2v_loss
                            else:
                                for k_vcls, v_vcls in zip(last_visual_clster.keys(), last_visual_clster.values()):
                                    if torch.equal(last_visual_clster[k_vcls], torch.zeros([1, 64]).cuda()):
                                        pass
                                    else:
                                        visual_clster_list.append(v_vcls)
                                        sematic_clster_list.append(last_sematic_clster[k_vcls])

                                if len(visual_clster_list) != 0:
                                    seen_visual_clster = torch.mean(fake_feature_train, dim=0, keepdim=True)
                                    seen_sematic_clster = s_cluster # s_cluster

                                    visual_clster_matrix = torch.cat([seen_visual_clster, (torch.cat(visual_clster_list, dim=0)).detach()], dim=0).unsqueeze(0)
                                    sematic_clster_matrix = torch.cat([seen_sematic_clster, (torch.cat(sematic_clster_list, dim=0))], dim=0).unsqueeze(0)

                                    visual_dist = (rel_distance(visual_clster_matrix, visual_clster_matrix, pc_distance=True)).squeeze(0)
                                    semantic_dist = (rel_distance(sematic_clster_matrix, sematic_clster_matrix, pc_distance=True)).squeeze(0)

                                    relation_loss = ((1 - F.cosine_similarity(semantic_dist, visual_dist, dim=-1))/2).sum()

                                    g_loss = self.criterion_generator(
                                            fake_features_cur + s_protos,
                                            real_features_pos, )
                                    total_loss = g_loss + contrast_loss + s2v_loss + relation_loss * 0.4
                                else:
                                    g_loss = self.criterion_generator(
                                            fake_features_cur + s_protos,
                                            real_features_pos, )
                                    total_loss = g_loss + contrast_loss + s2v_loss

                                generator_loss_sample += g_loss.item()
                                total_loss.backward()
                                self.optimizer_generator.step()
                        elif idx_in in args.unseen_classes_idx_metric[1:]:
                            if fake_feature_train.shape[0] > 100:
                                if torch.equal(cur_visual_clster[int(idx_in)], torch.zeros([1, 64]).cuda()):
                                    cur_visual_clster[int(idx_in)] = torch.mean(fake_feature_train, dim=0, keepdim=True)
                                else:
                                    cur_visual_clster[int(idx_in)] = cur_visual_clster[int(idx_in)] * args.r + torch.mean(fake_feature_train, dim=0, keepdim=True) * (1 - args.r)
                                cur_sematic_clster[int(idx_in)] = s_cluster
                            else:
                                pass

                        fake_features_i[idx_class] = fake_feature_train.clone() + s_protos.clone()

                generator_loss_batch += generator_loss_sample / len(unique_class)

                if args.real_seen_features and not has_unseen_class:
                    fake_features[count_sample_i] = real_features_i
                else:
                    fake_features[count_sample_i] = fake_features_i

            target = torch.squeeze(target)
            # ===================classification=====================
            self.optimizer.zero_grad()
            fake_features = fake_features.permute(0, 2, 1).contiguous()
            output = self.model.module.training_generative(fake_features.detach())
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            # ===================log=====================
            tbar.set_description(
                " G loss: {:.3f}".format(generator_loss_batch)
                + " C loss: {:.3f}".format(train_loss / (i + 1))
            )
            self.writer.add_scalar(
                "train/total_loss_iter", loss.item(), i + num_img_tr * epoch
            )

        self.writer.add_scalar("train/total_loss_epoch", train_loss, epoch)
        print("[Epoch: %d, numPCs: %5d]"% (epoch, i * self.args.batch_size + pts.data.shape[0]))
        print("Loss: {:.3f}".format(train_loss))

    def validation(self, epoch, args):
        print("IMPORTANT: This validation function is only for MONITORING (see Scannet folder for validation function)")
        print("Fast validation (only for monitoring during training):")
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc="\r")
        test_loss = 0.0

        i = None
        for i, sample in enumerate(tbar):
            features, pts, target, net_ids, net_pts, _, _ = get_data(sample, attributes=True, device="cuda")

            with torch.no_grad():
                output = self.model.module.forward(x=features, pos=pts, support_points=net_pts,\
                    indices=net_ids, gen_forward=True, backbone=False)
                output = output.permute(0, 2, 1).contiguous()
                loss = self.criterion(output.view(-1, self.N_CLASSES), target.view(-1))
                pred = softmax(output, dim=2).data.cpu().numpy()
                bias_matrix = np.zeros(pred.shape)
                bias_matrix[:, :, args.seen_classes_idx_metric] = 1 * args.bias
                pred = np.argmax(pred - bias_matrix, axis=2)

            test_loss += loss.item()
            tbar.set_description("Test loss: %.3f" % (test_loss / (i + 1)))
            target = target.cpu().numpy()
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        self.validation_metric_logs(test_loss=test_loss, epoch=epoch, num=i * self.args.batch_size + pts.data.shape[0])

if __name__ == "__main__":
    main(("sn"))