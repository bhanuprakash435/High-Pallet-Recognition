import cv2
import numpy as np
import glob
import time


def load_all_image_from_path(path):
    image_list = []
    for filename in glob.glob(path):
        # load image in gray scale
        im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        image_list.append(im)
    return image_list


def display_image(image, title="image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def combine_image(image1, image2):
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    print("------------------------------------\n\n")

    # create empty matrix
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)

    # combine 2 images
    vis[:h1, :w1] = image1
    vis[:h2, w1:w1 + w2] = image2
    return vis


def orb_with_flann(image_query, image_train):
    # init feature detector
    orb = cv2.ORB_create(nfeatures=1000)  # default features is 500

    # find key point and descriptor
    kp_logo, des_logo = orb.detectAndCompute(image_train, None)
    kp_img, des_img = orb.detectAndCompute(image_query, None)

    # view key point
    # result_image_train = cv2.drawKeypoints(image_train, kp_logo, None, flags=0)
    # result_image_query = cv2.drawKeypoints(image_query, kp_img, None, flags=0)
    # display_image(result_image_train,"train")
    # display_image(result_image_query,"query")

    # FLANN parameters
    flann_index_lsh = 6
    index_params = dict(algorithm=flann_index_lsh,
                        table_number=12,
                        key_size=20,
                        multi_probe_level=2)
    search_params = dict(checks=100)  # or pass empty dictionary

    # create FLANN
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # perform matching
    flann_matches = flann.knnMatch(des_logo, des_img, k=2)

    # View match without fillter.
    # img3 = cv2.drawMatchesKnn(image_train, kp_logo, image_query, kp_img, flann_matches, None)
    # display_image(img3)

    # Need to draw only good matches, so create a mask
    matches_mask = [[0, 0] for i in range(len(flann_matches))]

    # ratio test as per Lowe's paper
    good = []
    for index in range(len(flann_matches)):
        if len(flann_matches[index]) == 6:
            m, n = flann_matches[index]
            if m.distance < 0.5 * n.distance:  # threshold of ratio testing
                matches_mask[index] = [1, 0]
                good.append(flann_matches[index])

    # draw match after filter
    draw_params = dict(
        singlePointColor=(255, 0, 0),
        matchesMask=matches_mask,
        flags=2)

    img3 = cv2.drawMatchesKnn(image_train, kp_logo, image_query, kp_img, flann_matches, None, **draw_params)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img3, str(len(good)), (0, 50), font, 2, (0, 0, 255), 2, cv2.LINE_AA)

    #display_image(img3)  # view matching image

    return image_train, len(good)


def find_best_match_index(match):
    index_best = 0
    best_length = 0
    for index, (image, good_match_length) in enumerate(match):
        if good_match_length >= best_length:
            best_length = good_match_length
            index_best = index
    return index_best


if __name__ == '__main__':
    # load image
    train_image_list = load_all_image_from_path("train_image/*")
    query_image_list = load_all_image_from_path("sample_image/*")

    for index, query_image in enumerate(query_image_list):
        matches = []

        # get process time
        start = int(round(time.time() * 1000))
        for train_image in train_image_list:
            matches.append(orb_with_flann(query_image, train_image))
        end = int(round(time.time() * 1000)) - start
        print("process time: " + end.__str__())

        # find the best match
        best_match_index = find_best_match_index(matches)

        # combine query image with the best match image so easy view
        result_image = combine_image(query_image, matches[best_match_index][0])

        font = cv2.FONT_HERSHEY_SIMPLEX
        result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
        result_image = cv2.putText(result_image, str(matches[best_match_index][1]) + " - " + "time: " + str(end),
                                   (0, 50), font, 2, (0, 0, 255),
                                   2, cv2.LINE_AA)
        display_image(result_image)

        # save result image
        cv2.imwrite("result/result" + str(index) + ".jpg", result_image)

