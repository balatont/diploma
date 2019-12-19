import os
import numpy as np
from PIL import Image
from keras.utils import Sequence
#from skimage.io import imread


np.random.seed(seed=12345)

def load_data(nr_of_channels=3, batch_size=1, nr_speaker1_imgs=None, nr_speaker2_imgs=None,
              speaker1_train_ratio=0.6, speaker2_train_ratio=0.5, speaker1='s1', speaker2='s2',
              video_loc='../videos', speaker1_cached=True, speaker2_cached=True, save_speakers=True,
              generator=False):

    speaker1_image_names = load_data_from_speaker(video_loc, speaker1, nr_speaker1_imgs, speaker1_cached, save_speakers)
    speaker2_image_names = load_data_from_speaker(video_loc, speaker2, nr_speaker2_imgs, speaker2_cached, save_speakers)

    print("Speaker ", speaker1)
    (sp1_train_names, sp1_test_names) = split_speaker_data(speaker1_image_names, speaker1_train_ratio)
    print("Speaker ", speaker2)
    (sp2_train_names, sp2_test_names) = split_speaker_data(speaker2_image_names, speaker2_train_ratio)

    print("Data_nums:", len(sp1_train_names), len(sp2_train_names), len(sp1_test_names), len(sp2_test_names))

    if generator:
        return (data_sequence(sp1_train_names, sp2_train_names,
                             batch_size=batch_size, num_of_channels=nr_of_channels),
                data_sequence(sp1_test_names, sp2_test_names,
                             batch_size=batch_size, num_of_channels=nr_of_channels))
    else:
        print("Loading train images for speaker", speaker1)
        train1_images = create_image_array(sp1_train_names, nr_of_channels)
        print("Loading train images for speaker", speaker2)
        train2_images = create_image_array(sp2_train_names, nr_of_channels)
        print("Loading test images for speaker", speaker1)
        test1_images = create_image_array(sp1_test_names, nr_of_channels)
        print("Loading test images for speaker", speaker2)
        test2_images = create_image_array(sp2_test_names, nr_of_channels)

        print("Shapes:", train1_images.shape, train2_images.shape, test1_images.shape, test2_images.shape)

        return {"train1_images": train1_images, "train2_images": train2_images,
                "test1_images": test1_images, "test2_images": test2_images,
            "sp1_train_names": sp1_train_names,
            "sp2_train_names": sp2_train_names,
            "sp1_test_names": sp1_test_names,
            "sp2_test_names": sp2_test_names}


def load_data_from_speaker(video_loc, speaker, number_of_data, speaker_cached, save_speaker):
    speaker_really_cached = speaker_cached and os.path.isfile(os.path.join(video_loc, speaker, speaker + '.npy'))
    if speaker_really_cached:
        print("Loading data of speaker", speaker)
        image_names = np.load(video_loc + '/' + speaker + '/' + speaker + '.npy', allow_pickle=True)
        print("Finished loading data of speaker", speaker)
    else:
        print("Getting data from file system of speaker", speaker)
        speaker_path = os.path.join(video_loc, speaker)
        video_image_list = os.listdir(speaker_path)
        image_names = []
        for elem in video_image_list:
            if os.path.isdir(os.path.join(video_loc, speaker, elem)):
                tmp_list = []
                for file_name in os.listdir(os.path.join(video_loc, speaker, elem)):
                    if os.path.isfile(os.path.join(video_loc, speaker, elem, file_name)) and file_name[-1] == 'g':
                        tmp_list.append(video_loc + '/' + speaker + '/' + elem + '/' + file_name)
                np.array(tmp_list)
                image_names.append(tmp_list)

        image_names = np.array(image_names)

    if save_speaker and not speaker_really_cached:
        print("Saving data of speaker", speaker)
        np.save(video_loc + '/' + speaker + '/' + speaker + '.npy', image_names)
        print("Finished saving data of speaker", speaker)

    if number_of_data != None:
        image_names = image_names[:number_of_data]
    return image_names


def split_speaker_data(speaker_image_names, train_test_ratio):
    nr_data = len(speaker_image_names)

    np.random.shuffle(speaker_image_names)
    train_data_names = np.array(speaker_image_names[:int(train_test_ratio*nr_data)])
    test_data_names = np.array(speaker_image_names[int(train_test_ratio*nr_data):])

    train_names = np.array([val for sublist in train_data_names for val in sublist])
    test_names = np.array([val for sublist in test_data_names for val in sublist])

    return (train_names, test_names)


def create_image_array(image_list, nr_of_channels):
    image_array = []
    for image_name in image_list:
        if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
            if nr_of_channels == 1:  # Gray scale image -> MR image
                image = np.array(Image.open(image_name))
                image = image[:, :, np.newaxis]
            else:                   # RGB image -> street view
                image = np.array(Image.open(image_name))

            image = normalize_array(image)
            image_array.append(image)

    return np.array(image_array)

# If using 16 bit depth images, use the formula 'array = array / 32767.5 - 1' instead
def normalize_array(array):
    array = array / 127.5 - 1
    return array



class data_sequence(Sequence):

    def __init__(self, sp1_train_names, sp2_train_names, batch_size=1, num_of_channels=3):
        self.batch_size = batch_size
        self.num_of_channels = num_of_channels
        self.train_A = sp1_train_names
        self.train_B = sp2_train_names

    def __len__(self):
        return int(max(len(self.train_A), len(self.train_B)) / float(self.batch_size))

    def __getitem__(self, idx):  # , use_multiscale_discriminator, use_supervised_learning):if loop_index + batch_size >= min_nr_imgs:
        if idx >= min(len(self.train_A), len(self.train_B)):
            # If all images soon are used for one domain,
            # randomly pick from this domain
            if len(self.train_A) <= len(self.train_B):
                indexes_A = np.random.randint(len(self.train_A), size=self.batch_size)
                batch_A = []
                for i in indexes_A:
                    batch_A.append(self.train_A[i])
                batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]
            else:
                indexes_B = np.random.randint(len(self.train_B), size=self.batch_size)
                batch_B = []
                for i in indexes_B:
                    batch_B.append(self.train_B[i])
                batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]

        real_images_A = create_image_array(batch_A, self.num_of_channels)
        real_images_B = create_image_array(batch_B, self.num_of_channels)

        return real_images_A, real_images_B  # input_data, target_data



if __name__ == '__main__':
    load_data(nr_speaker1_imgs=20, nr_speaker2_imgs=20)
    #load_data(nr_speaker1_imgs=4, nr_speaker2_imgs=4)
