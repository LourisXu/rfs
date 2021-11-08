import numpy as np
import os
import cv2
import pickle


def load_data(data_dir, class_dirs, numEachClass=10):
    imgs = []
    lables = []

    for idx, class_dir in enumerate(class_dirs):
        cnt = 0
        for img_path in os.listdir(os.path.join(data_dir, class_dir)):
            img = cv2.imread(os.path.join(data_dir, class_dir, img_path), 1)
            # print(img.shape)
            imgs.append(img)
            lables.append(idx)
            # only load the specific number of imgs
            cnt += 1
            if cnt == numEachClass:
                break

    return np.array(imgs), lables


if __name__ == "__main__":
    data_dir = os.path.join("D:\\", "download", "dataset", "NCT-CRC-HE-100K-NONORM")

    class_dirs = os.listdir(os.path.join(data_dir))

    print(class_dirs)

    # x = list(np.arange(0, 10))
    # print(x)
    # y = [2, 3, 4]
    # z = np.setxor1d(x, y)
    # print(z)

    np.random.shuffle(class_dirs)
    print(class_dirs)

    train_classes, val_classes, test_classes = 3, 3, 3

    train_dirs = class_dirs[:train_classes]
    val_dirs = class_dirs[train_classes: train_classes + val_classes]
    test_dirs = class_dirs[train_classes + val_classes:]

    print(train_dirs, val_dirs, test_dirs)

    train_imgs, train_labels = load_data(data_dir, train_dirs)
    val_imgs, val_labels = load_data(data_dir, val_dirs)
    test_imgs, test_labels = load_data(data_dir, test_dirs)

    trainval_imgs = np.concatenate([train_imgs, val_imgs], axis=0)
    trainval_labels = []
    trainval_labels.extend(train_labels)
    trainval_labels.extend(val_labels)

    print(train_imgs.shape, len(train_labels))
    print(val_imgs.shape, len(val_labels))
    print(test_imgs.shape, len(test_labels))
    print(trainval_imgs.shape, len(trainval_labels))
    print(train_labels)

    save_dir = os.path.join('..', 'data', 'customDataset')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_dict = {'data': train_imgs, 'labels': train_labels};
    val_dict = {'data': val_imgs, 'labels': val_labels}
    test_dict = {'data': test_imgs, 'labels': test_labels}
    trainval_dict = {'data': trainval_imgs, 'labels': trainval_labels}

    f = open(os.path.join(save_dir, 'train.pickle'), 'wb')
    pickle.dump(train_dict, f)
    f.close()

    f = open(os.path.join(save_dir, 'val.pickle'), 'wb')
    pickle.dump(val_dict, f)
    f.close()

    f = open(os.path.join(save_dir, 'test.pickle'), 'wb')
    pickle.dump(test_dict, f)
    f.close()

    f = open(os.path.join(save_dir, 'trainval.pickle'), 'wb')
    pickle.dump(trainval_dict, f)
    f.close()
