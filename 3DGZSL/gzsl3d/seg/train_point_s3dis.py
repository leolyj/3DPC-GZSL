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

sys.path.append('..')
#Imports from 3DGZSL modules
from utils.loss import SegmentationLosses, InfoNCELosses
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from class_names import  CLASSES_NAMES_S3DIS
from dataloaders import make_data_loader
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

def get_model(model_name="SegBig", input_channels=3, output_channels=13, return_ll=False, args=None):
    if model_name == "SegBig":
        from gzsl3d.convpoint.networks.network_seg_orig import SegBig as Net
    return Net(input_channels, output_channels, return_ll=return_ll, args=args, w2v_size=args.w2c_size)

class Trainer(Trainer_default):

    def __init__(self, args):
        super().__init__(args)

        # Define Dataloader
        self.N_CLASSES = 13
        self.class_names = CLASSES_NAMES_S3DIS
        print("Create filelist...", end="")
        filelist_train = []
        filelist_test = []
        for area_idx in range(1, 7):
            folder = os.path.join(args.rootdir, "Area_{}".format(area_idx))
            datasets = [os.path.join("Area_{}".format(area_idx), dataset) for dataset in os.listdir(folder)]
            if area_idx == args.area:
                filelist_test = filelist_test + datasets
            else:
                filelist_train = filelist_train + datasets

        filelist_train.sort()
        filelist_test.sort()
        filelist_val = filelist_test.copy()
        (self.train_loader, self.val_loader, _, self.nclass,) = make_data_loader(
            args, flist_train=filelist_train, flist_val=filelist_val
            )

        if args.nocolor:
            model = get_model(model_name="SegBig", input_channels=1, output_channels=self.N_CLASSES, return_ll=False, args=args)
        else:
            model = get_model(model_name="SegBig", input_channels=3, output_channels=self.N_CLASSES, return_ll=False, args=args)

        # Load the ConvBasic version
        print("Load the pre-trained network from {}".format(os.path.join(args.savedir, "state_dict.pth")))
        model.load_state_dict(torch.load(os.path.join(args.savedir, "state_dict.pth")), strict=False)
        # Freeze all layers besides the linear layer that gets finetuned
        model.freeze_backbone_fcout()
        model.freeze_bn()

        train_params = [
            {"params": model.get_1x_lr_params(), "lr": args.lr},
            {"params": model.get_10x_lr_params(), "lr": args.lr * 10},
        ]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Generator and generator optimizer
        embed_feature_size = 0
        self.set_generator(embed_feature_size=embed_feature_size)

        class_weight = torch.ones(self.nclass)
        class_weight[args.unseen_classes_idx_metric] = args.unseen_weight

        if args.cuda:
            class_weight = class_weight.cuda()
        self.criterion = SegmentationLosses(weight=class_weight, cuda=args.cuda, bs=self.args.batch_size).build_loss(mode=args.loss_type)
        self.cst_criterion = InfoNCELosses(temperature=args.temperature, negative_mode='unpaired')

        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass, args.seen_classes_idx_metric, args.unseen_classes_idx_metric)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model = self.model.cuda()
            self.generator = self.generator.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            if args.cuda:
                self.model.load_state_dict(checkpoint["state_dict"])

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

        self.s2vmapper = nn.Linear(args.embed_dim, 128).cuda()

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
            if len(sample[0]) > 1:
                pts, features, target, embedding = (
                    sample[0], sample[1], sample[2], sample[3]
                )
                target = target[:, :, None]
                if self.args.cuda:
                    pts, features, target, embedding = (
                        pts.cuda(),
                        features.cuda(),
                        target.cuda(),
                        embedding.cuda(),
                    )
                self.scheduler(self.optimizer, i, epoch, self.best_pred)
                # ===================real feature extraction=====================
                with torch.no_grad():
                    real_features, _ = self.model.module.backbone(
                        x=features, input_pts=pts, return_mid=True
                    )
                    target_predicted_backbone = torch.argmax(self.model.module.trained_ss_ll(real_features, return_features=False), 2)
                    target_predicted_backbone = target_predicted_backbone.permute(1, 0).contiguous()
                real_features = real_features.permute(0, 2, 1)

                target_predicted_backbone = target_predicted_backbone[:, :, None]
                # ===================fake feature generation=====================
                fake_features = torch.zeros(real_features.shape)
                if args.cuda:
                    fake_features = fake_features.cuda()
                generator_loss_batch = 0.0

                for (count_sample_i, (real_features_i, target_i, embedding_i, target_predicted_i)) in enumerate(zip(real_features, target, embedding, target_predicted_backbone)):
                    generator_loss_sample = 0.0
                     ## reduce to real feature size
                    real_features_i = (
                        real_features_i.permute(1, 0)
                        .contiguous()
                        .view((-1, args.feature_dim))
                    )

                    target_i = target_i.view(-1)
                    target_predicted_i = target_predicted_i.view(-1)
                    fake_features_i = torch.zeros(real_features_i.shape)
                    if args.cuda:
                        fake_features_i = fake_features_i.cuda()

                    unique_class = torch.unique(target_i)

                    # test if image has unseen class point, if yes means no training for generator and generated features for the whole PC
                    has_unseen_class = False
                    for u_class in unique_class:
                        if u_class in args.unseen_classes_idx_metric:
                            has_unseen_class = True

                    if i != last_i:
                        cur_visual_clster = {}
                        cur_sematic_clster = {}
                        for unseen in args.unseen_classes_idx_metric:
                            cur_visual_clster[int(unseen)] = torch.zeros([1, 128]).cuda()
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
                            s_cluster = embedding_class[0].unsqueeze(0)

                            # Noise generation
                            z = torch.rand((embedding_class.shape[0], args.noise_dim))
                            if args.cuda:
                                z = z.cuda()

                            # Generation of the fake Features
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
                                ## in order to avoid CUDA out of memory
                                random_idx = torch.randint(
                                    low=0,
                                    high=fake_feature_train.shape[0],
                                    size=(args.batch_size_generator,),
                                )
                                n_protos = max(int(real_features_class.shape[0] * args.proto_ratio), 1)
                                visual_protos = self.getMutiplePrototypes(real_features_class,
                                                                          n_protos)  # [n_protos, 128]

                                s2v_loss = 0
                                for v in range(visual_protos.shape[0]):
                                    s2v_loss += (1 - F.cosine_similarity(visual_protos[v, :].unsqueeze(0), s_protos,
                                                                         dim=1)) / 2
                                s2v_loss = s2v_loss / (visual_protos.shape[0])

                                real_features_pos = real_features_class[random_idx]
                                fake_features_cur = fake_feature_train[random_idx]

                                _, D = real_features_pos.shape
                                neg_point_features = neg_point_features.reshape(-1, D)

                                contrast_loss = self.cst_criterion(fake_features_cur, real_features_pos,
                                                                   neg_point_features)
                                contrast_loss.requires_grad_(requires_grad=True)

                                visual_clster_list = []
                                sematic_clster_list = []
                                if i == 0 and count_sample_i == 0:
                                    g_loss = self.criterion_generator(
                                        fake_features_cur + s_protos,
                                        real_features_pos, )
                                    total_loss = g_loss + contrast_loss + s2v_loss
                                else:
                                    for k_vcls, v_vcls in zip(last_visual_clster.keys(), last_visual_clster.values()):
                                        if torch.equal(last_visual_clster[k_vcls], torch.zeros([1, 128]).cuda()):
                                            pass
                                        else:
                                            visual_clster_list.append(v_vcls)
                                            sematic_clster_list.append(last_sematic_clster[k_vcls])

                                    if len(visual_clster_list) != 0:
                                        seen_visual_clster = torch.mean(fake_feature_train, dim=0, keepdim=True)
                                        seen_sematic_clster = s_cluster

                                        visual_clster_matrix = torch.cat(
                                            [seen_visual_clster, (torch.cat(visual_clster_list, dim=0)).detach()],
                                            dim=0).unsqueeze(0)
                                        sematic_clster_matrix = torch.cat(
                                            [seen_sematic_clster, (torch.cat(sematic_clster_list, dim=0))],
                                            dim=0).unsqueeze(0)

                                        visual_dist = (rel_distance(visual_clster_matrix, visual_clster_matrix,
                                                                    pc_distance=True)).squeeze(0)
                                        semantic_dist = (rel_distance(sematic_clster_matrix, sematic_clster_matrix,
                                                                      pc_distance=True)).squeeze(0)

                                        relation_loss = ((1 - F.cosine_similarity(semantic_dist,
                                                                                  visual_dist,
                                                                                  dim=-1)) / 2).sum()

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
                            elif idx_in in args.unseen_classes_idx_metric:
                                if fake_feature_train.shape[0] > 100:
                                    if torch.equal(cur_visual_clster[int(idx_in)], torch.zeros([1, 128]).cuda()):
                                        cur_visual_clster[int(idx_in)] = torch.mean(fake_feature_train, dim=0,
                                                                                    keepdim=True)
                                    else:
                                        cur_visual_clster[int(idx_in)] = cur_visual_clster[
                                                                             int(idx_in)] * args.r + torch.mean(
                                            fake_feature_train, dim=0, keepdim=True) * (1 - args.r)
                                    cur_sematic_clster[int(idx_in)] = s_cluster
                                else:
                                    pass

                            fake_features_i[idx_class] = fake_feature_train.clone() + s_protos.clone()
                    generator_loss_batch += generator_loss_sample / len(unique_class)

                    if args.real_seen_features and not has_unseen_class:
                        fake_features[count_sample_i] = real_features_i.permute(1, 0)
                    else:
                        fake_features[count_sample_i] = fake_features_i.permute(1, 0)

                # ===================classification=====================
                self.optimizer.zero_grad()
                fake_features = fake_features.permute(0, 2, 1).contiguous()
                output = self.model.module.training_generative(
                    fake_features.detach(), npoints=pts.size()[1]
                )

                loss = self.criterion(output.view(-1, self.N_CLASSES), target.view(-1))
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
                self.writer.add_scalar(
                    "train/generator_loss", generator_loss_batch, i + num_img_tr * epoch
                )

        self.writer.add_scalar("train/total_loss_epoch", train_loss, epoch)
        print(
            "[Epoch: %d, numPCs: %5d]"
            % (epoch, i * self.args.batch_size + pts.data.shape[0])
        )
        print("Loss: {:.3f}".format(train_loss))

    def validation(self, epoch, args):
        # Important: This validation function only is a rough and random picking method for monitoring purposes during training.
        # A consistent and complete test/valiadtion has to be run with the method provided in the ConvPoint folder.
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc="\r")
        test_loss = 0.0
        i = None
        for i, sample in enumerate(tbar):
            pts, features, target, embedding = (
                sample[0], sample[1], sample[2], sample[3]
            )
            if self.args.cuda:
                pts, target, features, embedding = pts.cuda(), target.cuda(), features.cuda(), embedding.cuda()
            with torch.no_grad():
                output = self.model.module.forward_generative(x=features, input_pts=pts, return_features=False)
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
        self.validation_metric_logs(test_loss=test_loss, epoch=epoch, num=i * self.args.batch_size + pts.data.shape[0])

        print("IMPORTANT: This validation function is only for MONITORING (see ConvPoint folder for test function)")
        self.evaluator.reset()

if __name__ == "__main__":
    main("s3dis")
