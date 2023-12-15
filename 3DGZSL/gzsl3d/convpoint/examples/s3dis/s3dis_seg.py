# S3DIS Example with ConvPoint
# add the parent folder to the python path to access convpoint library
import argparse
import os
import random
import time
import numpy as np

from datetime import datetime
from tqdm import tqdm
from PIL import Image

import torch
import torch.utils.data
from torchvision import transforms
from pathlib import Path

import gzsl3d.convpoint.convpoint.knn.lib.python.nearest_neighbors as nearest_neighbors
from gzsl3d.word_representations.word_representations_utils import get_word_vector

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# wrap blue / green
def wblue(str):
    return bcolors.OKBLUE+str+bcolors.ENDC
def wgreen(str):
    return bcolors.OKGREEN+str+bcolors.ENDC


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def nearest_correspondance(pts_src, pts_dest, data_src, K=1):
    indices = nearest_neighbors.knn(pts_src.astype(np.float32), pts_dest.astype(np.float32), K, omp=True)
    if K == 1:
        indices = indices.ravel()
        data_dest = data_src[indices]
    else:
        data_dest = data_src[indices].mean(1)
    return data_dest

def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1],])
    return np.dot(batch_data, rotation_matrix)



# Part dataset only for training / validation
class PartDatasetTrainVal():

    def __init__(self, filelist, folder, training=False, block_size=2,
                 npoints=4096, iteration_number=None, nocolor=False,
                 jitter=0.4, attribute="w2v", use_unseen_seen=False,
                 unseen_idx_list=[4, 5, 7, 10], no_attribute=False):

        self.training = training
        self.filelist = filelist
        self.folder = folder
        self.bs = block_size
        self.nocolor = nocolor
        self.no_attribute = no_attribute
        self.npoints = npoints
        self.iterations = iteration_number
        self.verbose = False
        self.number_of_run = 10
        self.jitter = jitter
        self.transform = transforms.ColorJitter(
            brightness=jitter,
            contrast=jitter,
            saturation=jitter
        )
        self.unseen_idx_list = unseen_idx_list
        self.attribute = attribute
        self.use_unseen_seen = use_unseen_seen
        if not self.no_attribute:
            self.Glove_num_dict, self.w2v_num_dict, self.img_embd_array = get_word_vector(dataset="s3dis")

    def __getitem__(self, index):
        folder = self.folder
        if self.training:
            index = random.randint(0, len(self.filelist)-1)
            dataset = self.filelist[index]
        else:
            dataset = self.filelist[index//self.number_of_run]

        filename_data = os.path.join(folder, dataset, 'xyzrgb.npy')
        xyzrgb = np.load(filename_data).astype(np.float32)

        # load labels
        filename_labels = os.path.join(folder, dataset, 'label.npy')
        if self.verbose:
            print('{}-Loading {}...'.format(datetime.now(), filename_labels))
        labels = np.load(filename_labels).astype(int).flatten()

        # pick a random point
        select_unseen = True #Select as long
        rounds = 0
        while select_unseen:
            pt_id = random.randint(0, xyzrgb.shape[0]-1)
            pt = xyzrgb[pt_id, :3]

            mask_x = np.logical_and(xyzrgb[:, 0] < pt[0]+self.bs/2, xyzrgb[:, 0] > pt[0]-self.bs/2)
            mask_y = np.logical_and(xyzrgb[:, 1] < pt[1]+self.bs/2, xyzrgb[:, 1] > pt[1]-self.bs/2)
            mask = np.logical_and(mask_x, mask_y)
            pts = xyzrgb[mask]
            lbs = labels[mask]

            #Take also the random points --> such that perhaps also pillars are accepted which have only very few unseen labels
            choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
            pts = pts[choice]
            lbs = lbs[choice]

            unique_elements = np.unique(lbs)
            rounds += 1

            if not self.use_unseen_seen and (self.training and any(x in unique_elements.tolist() for x in self.unseen_idx_list)):
                select_unseen = True

            else:
                select_unseen = False

            if rounds > 500: #This defines how often it is tried in one point cloud to find a pillar
                if not self.no_attribute:
                    return self.__getitem__(random.randint(0, len(self.filelist)-1))
                else:
                    return self.__getitem__(random.randint(0, len(self.filelist)-1))

        if self.nocolor:
            features = np.ones((pts.shape[0], 1))
        else:
            features = pts[:, 3:]

            if self.training and self.jitter > 0:
                features = features.astype(np.uint8)
                features = np.array(self.transform(Image.fromarray(np.expand_dims(features, 0))))
                features = np.squeeze(features, 0)

            features = features.astype(np.float32)
            features = features / 255 - 0.5

        pts = pts[:, :3]
        if self.training:
            pts = rotate_point_cloud_z(pts)
        #Generate the mebeddings on the fly
        if not self.no_attribute:
            if self.attribute == "w2v":
                attributes = np.array(list(map(lambda x: self.w2v_num_dict[x], np.squeeze(lbs.astype(np.float32)))), dtype=np.float32)

            elif self.attribute == "glove":
                attributes = np.array(list(map(lambda x: self.Glove_num_dict[x], np.squeeze(lbs.astype(np.float32)))), dtype=np.float32)

            elif self.attribute == "glove_w2v":
                w2v_gt_attributes = np.array(list(map(lambda x: self.w2v_num_dict[x], np.squeeze(lbs.astype(np.float32)))), dtype=np.float32)
                glove_gt_attributes = np.array(list(map(lambda x: self.Glove_num_dict[x], np.squeeze(lbs.astype(np.float32)))), dtype=np.float32)
                attributes = np.hstack((glove_gt_attributes, w2v_gt_attributes.astype(np.float32)))
            else:
                attributes = np.array(list(map(lambda x: self.img_embd_array[x], np.squeeze(lbs.astype(np.float32)))), dtype=np.float32)

        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(features).float()
        lbs = torch.from_numpy(lbs).long()
        if not self.no_attribute:
            attributes = torch.from_numpy(attributes).float()
            return pts, fts, lbs, attributes
        else:
            return pts, fts, lbs
    def __len__(self):
        if self.iterations is None:
            return len(self.filelist) * self.number_of_run
        else:
            return self.iterations

# Part dataset only for testing
class PartDatasetTest():

    def compute_mask(self, pt, bs):
        # build the mask
        mask_x = np.logical_and(self.xyzrgb[:, 0] <= pt[0]+bs/2, self.xyzrgb[:, 0] >= pt[0]-bs/2)
        mask_y = np.logical_and(self.xyzrgb[:, 1] <= pt[1]+bs/2, self.xyzrgb[:, 1] >= pt[1]-bs/2)
        mask = np.logical_and(mask_x, mask_y)
        return mask

    def __init__(self, filename, folder,
                 block_size=2, npoints=4096,
                 min_pick_per_point=1, test_step=0.5,
                 nocolor=False):

        self.folder = folder
        self.bs = block_size
        self.npoints = npoints
        self.verbose = False
        self.min_pick_per_point = min_pick_per_point
        self.nocolor = nocolor
        # load data
        self.filename = filename
        filename_data = os.path.join(folder, self.filename, 'xyzrgb.npy')
        if self.verbose:
            print('{}-Loading {}...'.format(datetime.now(), filename_data))
        self.xyzrgb = np.load(filename_data)
        filename_labels = os.path.join(folder, self.filename, 'label.npy')
        if self.verbose:
            print('{}-Loading {}...'.format(datetime.now(), filename_labels))
        self.labels = np.load(filename_labels).astype(int).flatten()

        step = test_step
        mini = self.xyzrgb[:, :2].min(0)
        discretized = ((self.xyzrgb[:, :2]-mini).astype(float)/step).astype(int)
        self.pts = np.unique(discretized, axis=0)
        self.pts = self.pts.astype(np.float)*step + mini + step/2

    def __getitem__(self, index):

        # get the data
        mask = self.compute_mask(self.pts[index], self.bs)
        pts = self.xyzrgb[mask]

        # choose right number of points
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]

        # labels will contain indices in the original point cloud
        lbs = np.where(mask)[0][choice]

        if self.nocolor:
            features = np.ones((pts.shape[0], 1))
        else:
            features = pts[:, 3:6] / 255 - 0.5
        pts = pts[:, :3].copy()

        # convert to torch
        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(features).float()
        lbs = torch.from_numpy(lbs).long()

        return pts, fts, lbs

    def __len__(self):
        # return len(self.pts)
        return self.pts.shape[0]


def get_model(model_name, input_channels, output_channels, return_ll, args):
    print("Model name {}".format(model_name))
    print("Input channels {}".format(input_channels))
    print("Output channels {}".format(output_channels))
    if model_name == "SegBig":
        #from networks.network_seg import SegBig as Net
        from gzsl3d.convpoint.networks.network_seg_orig import SegBig as Net
    return Net(input_channels, output_channels, return_ll=return_ll, w2v_size=args.attribute_size, args=args)


def test(args, flist_test):

    N_CLASSES = 13
    # create the network
    print("Creating network...")
    if args.nocolor:
        net = get_model(args.model, input_channels=1, output_channels=N_CLASSES, return_ll=True, args=args)
    else:
        net = get_model(args.model, input_channels=3, output_channels=N_CLASSES, return_ll=True, args=args)
    if args.use_zsl_head:

        print("Using the ZSL head, it is loaded the network from {}".format(args.zsl_trained_path))
        checkpoint = torch.load(args.zsl_trained_path)
        #Filter out the ones that don't match
        filter_dict = {"head_softmax.mlp.weight":0, "head_softmax.batch_norm.bias":0, "head_softmax_generative.mlp.weight":0,\
             "head_softmax_generative.batch_norm.bias":0, "fcout_baseline.weight":0, "fcout_baseline.bias":0}
        pretrained_filtered_dict = {k: v for k, v in checkpoint['state_dict'].items() if not k in filter_dict}

        net.load_state_dict(pretrained_filtered_dict, strict=False)
    else:
        print("Using the backbone head, it is loaded the network from {}".format(args.zsl_trained_path))
        checkpoint = torch.load(args.zsl_trained_path)
        #Filter out the ones that don't match
        filter_dict = {"head_softmax.mlp.weight":0, "head_softmax.batch_norm.bias":0, "head_softmax_generative.mlp.weight":0,\
             "head_softmax_generative.batch_norm.bias":0, "fcout_baseline.weight":0, "fcout_baseline.bias":0}
        pretrained_filtered_dict = {k: v for k, v in checkpoint['state_dict'].items() if not k in filter_dict}
        net.load_state_dict(checkpoint['state_dict'], strict=True)
    net.cuda()
    net.eval()
    print("parameters", count_parameters(net))
    for filename in flist_test:
        print(filename)
        label_path = os.path.join(args.rootdir, filename, "label.npy")
        print("{}".format(label_path))
        gt_label = np.load(label_path).astype(np.float32)
        print("GT size {}".format(gt_label.shape))
        print("GT label type {}".format(gt_label.dtype))
        ds = PartDatasetTest(filename, args.rootdir,
                             block_size=args.blocksize,
                             min_pick_per_point=args.npick,
                             npoints=args.npoints,
                             test_step=args.test_step,
                             nocolor=args.nocolor
                            )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batchsize, shuffle=False,
            num_workers=args.threads
        )

        xyzrgb = ds.xyzrgb[:, :3]
        print("XYZRGB {}".format(xyzrgb.shape))
        scores = np.zeros((xyzrgb.shape[0], N_CLASSES))
        number_predicted = np.zeros((xyzrgb.shape[0]))
        total_time = 0
        iter_nb = 0
        with torch.no_grad():
            t = tqdm(loader, ncols=80)
            for pts, features, indices in t:
                t1 = time.time()
                features = features.cuda()
                pts = pts.cuda()
                if args.use_zsl_head:

                    outputs = net.forward_generative(features, pts, return_features=False)
                    outputs_np = outputs.cpu().numpy().reshape((-1, N_CLASSES))

                else:
                    outputs, _ = net(features, pts)
                    outputs_np = outputs.cpu().numpy().reshape((-1, N_CLASSES))
                t2 = time.time()

                scores[indices.cpu().numpy().ravel()] += outputs_np
                number_predicted[indices.cpu().numpy().ravel()] = number_predicted[indices.cpu().numpy().ravel()] + 1

                iter_nb += 1
                total_time += (t2-t1)
                t.set_postfix(time="{:05e}".format(total_time/(iter_nb*args.batchsize)))


        mask = np.logical_not(scores.sum(1) == 0)
        scores = scores[mask]
        pts_src = xyzrgb[mask]

        # create the scores for all points
        scores = nearest_correspondance(pts_src, xyzrgb, scores, K=1)
        # compute softmax
        scores = scores - scores.max(axis=1)[:, None]
        scores = np.exp(scores) / np.exp(scores).sum(1)[:, None]
        scores = np.nan_to_num(scores)
        bias_matrix = np.zeros(scores.shape)
        bias_matrix[:, args.seen_classes_idx_metric] = 1 * args.bias
        os.makedirs(os.path.join(args.savedir, filename), exist_ok=True)

        # saving labels
        save_fname = os.path.join(args.savedir, filename, "pred.txt")

        #Bias values
        scores = scores - bias_matrix
        scores = scores.argmax(1)
        np.savetxt(save_fname, scores, fmt='%d')

        if args.savepts:
            save_fname = os.path.join(args.savedir, filename, "pts_scores_label.txt")
            xyzrgb = np.concatenate([xyzrgb, np.expand_dims(scores, 1), gt_label], axis=1)
            np.savetxt(save_fname, xyzrgb, fmt=['%.4f', '%.4f', '%.4f', '%d', '%d'])

        #Save all the files
        tmp_save_path = os.path.join(args.savedir, filename)
        Path(tmp_save_path).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(tmp_save_path, "label"), gt_label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ply", action="store_true", help="save ply files (test mode)")
    parser.add_argument("--savedir", default="results/", type=str)
    parser.add_argument("--rootdir", type=str)
    parser.add_argument("--batchsize", "-b", default=4, type=int) # Originally 16
    parser.add_argument("--npoints", default=8192, type=int)
    parser.add_argument("--area", default=1, type=int)
    parser.add_argument("--blocksize", default=2, type=int)
    parser.add_argument("--iter", default=1000, type=int)
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--npick", default=16, type=int)
    parser.add_argument("--savepts", action="store_true")
    parser.add_argument("--nocolor", action="store_true")
    parser.add_argument("--test_step", default=0.2, type=float)
    parser.add_argument("--nepochs", default=50, type=int)
    parser.add_argument("--jitter", default=0.4, type=float)
    parser.add_argument("--model", default="SegBig", type=str)
    parser.add_argument("--drop", default=0, type=float)
    parser.add_argument("--cluster", default=False, type=bool)
    parser.add_argument("--unseen_idx_list", default=[4, 5, 7, 10], type=list)
    parser.add_argument("--use_no_attribute", default=False, type=bool)
    parser.add_argument("--use_zsl_head", default=False, type=bool)
    parser.add_argument("--zsl_trained_path", type=str)
    parser.add_argument("--bias", type=float, default=0.4)
    parser.add_argument("--attribute_size", type=int, default=600) # 600
    parser.add_argument("--base_width", type=int, default=1)
    args = parser.parse_args()

    seen_classes_idx_metric = np.arange(13)
    seen_classes_idx_metric = np.delete(
        seen_classes_idx_metric, np.array(args.unseen_idx_list)
    )
    args.seen_classes_idx_metric = seen_classes_idx_metric
    args.rootdir = "../../../../data/s3dis/processed_data/"

    print("Args.savedir/data root {}".format(args.savedir))

    # create the filelits (train / val) according to area
    print("Create filelist...", end="")
    filelist_train = []
    filelist_test = []
    print("args rootdir {}".format(args.rootdir))
    for area_idx in range(1, 7):
        folder = os.path.join(args.rootdir, "Area_{}".format(area_idx))
        datasets = [os.path.join("Area_{}".format(area_idx), dataset) for dataset in os.listdir(folder)]
        if area_idx == args.area:
            filelist_test = filelist_test + datasets
        else:
            filelist_train = filelist_train + datasets
    filelist_train.sort()
    filelist_test.sort()
    print("done, {} train files, {} test files".format(len(filelist_train), len(filelist_test)))

    #count_pillars(args, filelist_train, filelist_test)
    if args.test:
        test(args, filelist_test)


if __name__ == '__main__':
    main()
