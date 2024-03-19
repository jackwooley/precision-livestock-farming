import os
import numpy as np
from random import randint
import pickle
import requests
# import seaborn as sns
from PIL import Image, ImageDraw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# clustering and dimension reduction
from sklearn.cluster import KMeans, DBSCAN
import re
import datetime
from matplotlib.colors import XKCD_COLORS

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import cv2
import torch
from torchvision import models, transforms

# utilty stuff to get the file to run on the command line
import sys
from typing import Union
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

### TODO -- have chatgpt make docstrings, comments, etc.
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main(video_source_folder_: str, output_image_folder_: str, crop_reg: list, nth_frame_: int):
    
    # Setting up all the inputs, etc.
    video_source_folder = video_source_folder_  # '20230920_videos/20230920-Dataset Bunk 3 videos'
    image_folder = output_image_folder_  # 'frames_from_videos/gt_frames_new_new'
    crop_regions = crop_reg  # [(180, 1, 450, 400)]  # use to crop out the feed bins; good for bin no. 3
    nth_frame = nth_frame_

    video_files_only = string_into_sorted_dir_list(video_source_folder, extension='.mp4')
    date = '2023-08-22' ### TODO -- create a function that parses the date out from the filenames, add date as an option on the command line maybe

    # for filet in video_files_only:
    #     video_processor(video_source_folder, image_folder, filet, crop_regions, date, nth_frame)
    
    # get a list of all the images sorted by image name (no path to them though)
    sorted_image_names = get_sorted_image_list(image_folder)
    sorted_image_paths = [os.path.join(image_folder, im_path) for im_path in sorted_image_names]  # add the path to the images
    
    # turn images from the recently processed video into a numpy array
    img_array = images_into_array(sorted_image_names, image_folder)  # gets sorted by image name inside the function
    
    # extract features from array of images
    resnet_features = extract_features(img_array)
    
    # reduce their dimensionality
    pca_reduced_dimensions_all_images = pca_(resnet_features, 2)
    
    # do kmeans to separate into two clusters
    km_labels, km_object = kmeans_(pca_reduced_dimensions_all_images, 2)
    
    # use binary classifier resnet to "decide" which cluster has a cow in it
    batch_size = 1
    class_names = ['has_a_cow', 'has_no_cow']
    binary_classifier = load_binary_classifier(class_names)
    for lab in np.unique(km_labels):
        # get all images that correspond to a certain cluster
        sampled_img_paths = get_images_from_indices(km_labels, lab, sorted_image_names, output_image_folder_)  
        # transform the images
        get_labels_from_these_images = InferenceDataset(sampled_img_paths, transform=resnet18_transform)
        # create dataloader from those images
        labels_dataloader = DataLoader(get_labels_from_these_images, batch_size=batch_size, shuffle=False)
        # get predictions from those images
        cow_presence_preds = identify_animal_cluster(binary_classifier, labels_dataloader, class_names)
        if sum(np.array(cow_presence_preds) == 'has_a_cow') >= (.8 * len(cow_presence_preds)):
            print(f'Many cow images detected in cluster {lab}. {lab} will be treated as the cluster with cows for the animal id section.')
            cluster_with_cows = lab
            break
        else:
            print(f'Few cow images detected in cluster {lab}.')
    feature_subset = np.array(resnet_features)[km_labels == cluster_with_cows][:,:,0,0]
    
    animal_ids = assign_animal_ids(feature_subset)
    return animal_ids
    
    ### TODO -- function for other transform (resnet18 input size instead of resnet50 size)
    
    ### TODO -- # get image_ids associated with each label in the kmeans
    
    # return video_source_folder, image_folder, crop_regions, nth_frame


def video_processor(video_source_folder: str, output_image_folder_: str, filename: str, crop_region_: tuple, date_: str, nth_frame: int):
    """Get frames from videos, crop them to just have the feedbunks, and write them to a specified location.
    
    Params:
    video_source_folder -- file location where the video is saved
    output_image_folder_ -- file location to write to
    filename -- name of the video file to process
    crop_region_ -- region specifying where and how to crop the image
    date_ -- date the videos were taken on
    nth_frame -- specifies how frequently a frame is captured (e.g., if nth_frame == 120, then every 120th frame is cropped and written). Should be >= 1
    """
    all_images_array = []
    print(filename)
    frame_number=0  # start the frames over at zero
    file_path = os.path.join(video_source_folder, filename)
    failed_iterations = {}

    # get all the video stuff set up
    video_capture, w, h, fourcc, fps = get_video_processing_parameters(file_path)  # homemade function

    # keep track of the timestamps
    timestamp1, timestamp2 = get_times_from_filenames(filename, date_)

    # loop through the video and get every nth frame
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3 - not sure if this is relevant anymore
        # make sure that the video capture worked correctly
        if ret != True:
            break
        if frame_number % nth_frame != 0:  # check if it's one of every n frames
            frame_number += 1  # increment if it is not one of the nth frames
            continue  # skip the file cropping/saving if modulo(frame_num, nth_frame) != 0 and continue on to the next iteration without doing anything else

        # if the frame *is* one of the nth frames, do the following
        try:
            image_to_write = crop_image(crop_region_, frame)
        except TypeError as te:
            print(f'An error occurred while cropping frame {timestamp1} from {filename} ({te}).')
            failed_iterations[filename] = [frame_number, 'crop']        
        try:
            # write_image(image_to_write, timestamp1, output_image_folder_) #(image, timestamp: datetime.datetime.strptime, output_image_folder: str
            all_images_array.append(image_to_write)
        # except TypeError as te:
        #     print(f'An error occurred while writing frame {timestamp1} from {filename} ({te}).')
        #     failed_iterations[filename] = [frame_number, 'write']
        except BaseException as be:
            print(f'An error occurred while appending {timestamp1} to all_images_array ({be}).')
            failed_iterations[filename] = [frame_number, 'append']

        timestamp1 = time_adder(timestamp1, nth_frame, fps=fps)  # plays the same role as unique_frame_counter
        if timestamp1 > timestamp2:
            break
        frame_number += 1  # even though you're using the timestamp to label images, keep this in so that you can make sure you're getting every nth frame without issues.

        if (frame_number - 1) % 150 == 0:
            print(f'Frame number {frame_number - 1} ({timestamp1}) was just processed.')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    print(f'Unsuccessful iterations: {list(failed_iterations.values())}')
    return all_images_array


def get_video_processing_parameters(file_path_: str, codec_string: str = 'XVID'):
    """Obtain video capture object, w/h/fps, and fourcc (which is a character code used to indicate the video codec I believe)."""
    try:
        video_capture = cv2.VideoCapture(file_path_)
        w = int(video_capture.get(3))  # width
        h = int(video_capture.get(4))  # height
        fourcc = cv2.VideoWriter_fourcc(*codec_string)
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        return video_capture, w, h, fourcc, fps
    except BaseException as be:
        raise BaseException(f'An error ({be}) occurred in the get_video_processing_parameters function. This might indicate an issue with how the video is being read, or an error in the filepath provided.')
    return


def crop_image(crop_region_: Union[tuple, list, np.array], frame):
    """Crop an image to a certain size.
    
    Params:
    crop_region -- should be an iterable with a region specified using four numbers, like the following: [x, y, w, h]  ### TODO -- explain more here'
    frame -- a frame from the video
    """
    for idx, crop_region in enumerate(crop_region_):
        x, y, w, h = crop_region_
        try:
            cropped_frame = frame[y:y + h, x:x + w]  # crop the frame
            img = np.uint8(cropped_frame)  # convert to np array
        except BaseException as b:  # update with a more specific error
            # failed_iterations[filet] = frame_number
            # print(f'something went wrong while trying to crop img_{frame_timestamp}: {b} ({type(b)})')
            pass
    
    return img


def write_image(image, timestamp: datetime.datetime.strptime, output_image_folder: str):
    """Write an image to a specified file location. 
    
    Params:
    image -- the image to write
    timestamp -- the timestamp corresponding to this frame; should be datetime.datetime.strptime type (still working on the type hint here)
    output_image_folder -- what it sounds like; where the image is written to
    """
    try:
        timestamp_as_string = re.sub('[^A-Za-z0-9]', '_', str(timestamp))
        cv2.imwrite(f'{output_image_folder}/img_{timestamp_as_string}.png', image)
    except BaseException as be:
        print(f'An error occurred in the write_image function: {be}')
    return


def string_into_sorted_dir_list(source_folder: str, extension: str = '.mp4'):
    """Access the video files of a given type in a directory specified by a given string and return a sorted list of them."""

    try:
        list_of_files_in_folder = os.listdir(source_folder)  # list all files in a directory
        new_list_of_files = []
        
        for filename in list_of_files_in_folder:
            if filename[-len(extension):] == extension:  # sort to only have files with the right extension
                new_list_of_files.append(filename)
        
        new_list_of_files.sort()
        return new_list_of_files
    except FileNotFoundError as fe:
        raise FileNotFoundError(f'{fe}: The folder you input ({source_folder}) cannot be found. Double-check to make sure you\'re in the right directory and that the name is spelled correctly.')


# get images with timestamps in their names
def get_times_from_filenames(filename: str, date_: str) -> tuple:
    """Create datetime objects for the beginning and end timestamps of a video.
    
    Params
    filename -- should have a substring that identifies it followed by two substrings that contain times of day (formatted HHMMSS), separated by hyphens
    date -- the date that the video was filmed on, formatted as %Y-%m-%d (YYYY-MM-DD)
    
    Returns
    tuple of length 2; each element has a date + time together in a datetime object
    """

    ### TODO -- verify inputs, make sure that date and filename are appropriately formatted
    if bool(re.match('\d{4}-\d{2}-\d{2}', date_)) & bool(re.match('.*-\d{6}-\d{6}', filename)):
        split_name = re.split('-', filename)
        first = date_ + ' ' + split_name[1]
        second = date_ + ' ' + split_name[2][0:6]
        first_dt = datetime.datetime.strptime(first, '%Y-%m-%d %H%M%S')
        second_dt = datetime.datetime.strptime(second, '%Y-%m-%d %H%M%S')
    else:
        raise ValueError('Either the filename or the date supplied was not formatted correctly. please check the inputs to make sure they conform to the accepted format in the docstring.')
        
    return (first_dt, second_dt)


def time_adder(datetime_object: datetime.datetime, every_n_frames: float, fps=30):
    """Add n seconds to a datetime object. 
    
    Params:
    datetime_object -- some datetime object ofc
    every_n_frames -- every nth frame you want to capture
    fps -- frames per second of the video you're working with; should basically always be 30
    
    Returns:
    datetime_object -- the original datetime object with (every_n_frames / fps) seconds added to it
    
    !!! ONLY USE FOR WORKING WITH n_seconds >= 1 (i.e., only when getting 1 frame per second or a slower rate) !!!
    """
    
    n_seconds = every_n_frames / fps
    if n_seconds >= 1:
        datetime_object += datetime.timedelta(seconds=n_seconds)
        return datetime_object
    else:
        raise ValueError('This function should not be used with (every_n_seconds / fps) < 1 due to rounding errors')


def get_sorted_image_list(image_folder_path):
    image_list = os.listdir(image_folder_path)
    image_list.sort()
    return image_list


def images_into_array(image_list, image_folder_path):  # image_folder_path should be the same as output_folder
    """Open all the images and append them together into a big numpy array."""
    
    # image_list = os.listdir(image_folder_path)
    # image_list.sort()
    
    images = []
    for file in image_list:
        # remove .ipynb_checkpoints
        if file[-4:] in ['.png', '.jpg']:
            file_path = os.path.join(image_folder_path, file)
            image = cv2.imread(file_path)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            
            image_data = np.array(images)
    
    return image_data  # should be a 4-dimensional tensor


def load_feature_extractor():
    """Load a resnet in evaluation mode."""
    
    resnet = models.resnet50(pretrained=True)
    resnet.eval()
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
    return feature_extractor


def resnet50_transform():
    """Create a transformation pipeline with the appropriate input image size for a resnet50."""
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


def resnet18_transform():
    """Create a transformation pipeline with the appropriate input image size for a resnet18."""
    ### TODO -- combine the two transformation functions into one that just takes the input dumension as an argument
    mean = np.array([0.485, 0.456, 0.406])  # double-check to make sure these are the right values
    std = np.array([0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform


def extract_features(image_array):
    """Using the resnet, extract features from a given image."""

    # load resnet50 in eval mode w last layer "sliced" off, create transform pipeline
    feature_extractor = load_feature_extractor()
    transform = resnet50_transform()
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Beginning feature extraction using {DEVICE}...')
    fe = feature_extractor.to(DEVICE)
    image_tensor = torch.from_numpy(image_array)#.to(DEVICE)
    
    all_features = []
    for img in image_tensor:
        img = transform(Image.fromarray(img.numpy()))
        img = img.unsqueeze(0)
        features = feature_extractor(img.to(DEVICE))
        all_features.append(features.detach().cpu().numpy())
        
    all_features = np.array(all_features)[:,0,:,0,0]    #make sure it's the right dimension
    
    return all_features  # should be len(image_array) x 2048


def tsne_(features: np.array, n_comps: int = 2):  ### TODO -- UPDATE THIS AND MAIN() SO THAT YOU CAN MANUALLY SPECIFY N_COMPS AND EXPERIMENT WITH IT
    tsne = TSNE(n_components=n_comps, random_state=42)
    tsne_result = tsne.fit_transform(features)
    return tsne_result


def pca_(features: np.array, n_comps: int = 2):  ### TODO -- UPDATE THIS AND MAIN() SO THAT YOU CAN MANUALLY SPECIFY N_COMPS AND EXPERIMENT WITH IT
    pca = PCA(n_comps, random_state=42)
    pca_result = pca.fit_transform(features)
    return pca_result


def kmeans_(features, n_clusts: int = 2):
    km = KMeans(n_clusters=n_clusts, random_state=42)
    cluster_labels = km.fit_predict(features)
    return cluster_labels, km  # return the kmeans object for plotting


def check_and_mkdir(directory_path: str):
    """Check if a directory for plots has been created and create it if it does not exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created. Plots and other artifacts will be saved here.") ### TODO -- add functionality to save pca, tsne, dbscan, kmeans models
    else:
        print(f"Directory '{directory_path}' already exists. Plots and other artifacts will be saved here.")


### TODO -- SAVE THE RAW IMAGE DATA/LABELS FOR KMEANS


def create_plots(km_, features):  # don't worry too much about this for now, and instead focus on returning the raw image name labels associated with their clusters.
    """Plot the different clusters generated by the k-Means algorithm and save them to a file location."""
    
    # check the file location and create if necessary
    check_and_mkdir('end_to_end_plots')
    
    # get the colors for plotting all set up
    xkcd_colors = list(XKCD_COLORS.values())
    xkcd_colors.sort()
    
    kmeans_labels = km_.labels_
    u_labels = np.unique(kmeans_labels)
    centroids = km_.cluster_centers_
    
    #plotting the results:
    fig, ax = plt.subplots(figsize=(12, 12))
    color_ = 0
    for i in range(len(u_labels)):
        if u_labels[i] == i:
            ax.scatter(features[kmeans_labels == i, 0] , features[kmeans_labels == i, 1], 
                       c=xkcd_colors[(len(xkcd_colors) // len(u_labels)) * color_])
        color_ += 1
    plt.savefig('end_to_end_plots/')
    return 


class InferenceDataset(Dataset):  # this needs a better name, ngl
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        image = Image.open(self.file_paths[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image
    

def load_binary_classifier(class_names):
    bc = models.resnet18(pretrained=False)  # load model
    # class_names = ['has_a_cow', 'has_no_cow']
    # adjust number of features to match the input
    num_features = bc.fc.in_features
    bc.fc = torch.nn.Linear(num_features, len(class_names))
    # load the fine-tuned version
    bc.load_state_dict(torch.load('binary_classifier_cnn.pth'))
    bc.to(DEVICE)  # send to GPU, if available
    return bc


def check_match(element_, label_number_):
    # return np.array(array_to_check) == label_number
    return element_ == label_number_


def indices_matching_condition(lst, label_number):
    return [index for index, element in enumerate(lst) if check_match(element, label_number)]


def get_images_from_indices(array_to_match_from, what_to_match, image_list_to_filter, path_to_images):
    """
    Params:
    array_to_match_from: the kmeans_labels array (or equivalent)
    what_to_match: one of the unique values in kmeans_labels (probably will be 0 or 1)
    image_list_to_filter: a list of image names
    path_to_images: the relative path to the images in image_list_to_filter
    """

    n_to_sample = int(.12 * sum(np.array(array_to_match_from) == what_to_match))
    # get all indices for a certain label
    indices = indices_matching_condition(array_to_match_from, what_to_match)

    # randomly sample from those indices
    indices_subset = random.sample(indices, n_to_sample)

    # get a list of the paths to the images in the subset
    image_path_subset = [os.path.join(path_to_images, path) for path in list(np.array(image_list_to_filter)[indices_subset])]
    
    return image_path_subset


def identify_animal_cluster(classifier, dataloader, class_names):
    classifier.eval()
    cow_present_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(DEVICE)
            outputs = classifier(batch)
            _, predicted_classes = torch.max(outputs, 1)
            
            cow_present_predictions.append(class_names[predicted_classes.cpu().numpy()[0]])
    return cow_present_predictions


def assign_animal_ids(feature_subset_):
    """Identify the animals using DBSCAN to group them into different clusters."""
    # reduce dimensions (is it overfit?)
    tt = TSNE(n_components=2, random_state=42)
    tt_results = tt.fit_transform(feature_subset_)
    
    # cluster w dbscan (overfitting watch!)
    dbscan = DBSCAN(eps=10, min_samples=2)
    animal_id_preds = dbscan.fit_predict(tt_results)
    return animal_id_preds
    

if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument(
        '--video_folder',  # TODO -- figure out how to include shortened flags maybe
        nargs=1,
        type=str
    )
    cli.add_argument(
        '--image_folder',
        nargs=1,
        type=str
    )
    cli.add_argument(
        '--region',
        nargs='+',
        type=int
    )
    cli.add_argument(
        '--nth',
        nargs=1,
        type=int
    )
    args = cli.parse_args()
    
    # python3 end_to_end.py --video_folder sandbox/videos_to_process/ --image_folder sandbox/write_to_here/ --region 180 1 450 400 --nth 120
    
    # print(args.source_folder[0], args.output_folder[0], args.region, args.nth[0])
    # video_source_folder = args[0]
    # output_image_folder = args[1]
    # crop_regions = args[2]
    # nth_frame_ = args[3]
    main(args.video_folder[0], args.image_folder[0], args.region, args.nth[0])
