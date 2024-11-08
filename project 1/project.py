import scipy.io 
import os
import numpy as np
import matplotlib.pyplot as plt


def normalization(data, mean_vector, std_vector):
    # samples = data['x'].reshape(len(data['x']), -1)
    normalized_samples = (data - mean_vector)/std_vector
    return normalized_samples

def PCA(data):
    cov_matrix = np.cov(data, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # first_pca = eigenvectors[:2]
    first_pca = eigenvectors[:, 0]
    second_pca = eigenvectors[:, 1]
    norm_frist_pca = first_pca / np.linalg.norm(first_pca)
    norm_second_pca = second_pca / np.linalg.norm(second_pca)

    return norm_frist_pca, norm_second_pca

def projection(first_pca, second_pca, data):
    dimensions = np.column_stack((first_pca, second_pca))
    data_pca_2d = data@dimensions
    return data_pca_2d

def view(data_pca_2d, original_data, name):
    label = original_data['y'][0]
    T_shirt = []
    Sneaker = []
    for i in range(len(label)):
        if(label[i] == 1):
            Sneaker.append(data_pca_2d[i])
        else:
            T_shirt.append(data_pca_2d[i])
    T_shirt = np.array(T_shirt)
    Sneaker = np.array(Sneaker)
    colors = np.array(['red', 'green'])
    point_colors = colors[label]

    plt.figure(figsize=(8, 6))
    plt.scatter(Sneaker[:, 0], Sneaker[:, 1], s=5, alpha=0.5,label='Sneaker',color='red')
    plt.scatter(T_shirt[:, 0], T_shirt[:, 1], s=5, alpha=0.5,label='T-shirt',color='blue')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Projection onto the First Two Principal Components in "+name)
    plt.grid(True)
    plt.show()
    return T_shirt, Sneaker

def MLE(data):
    mu_x, mu_y = np.mean(data, axis=0)
    mu = np.array([mu_x, mu_y])
    cov_matrix = np.cov(data, rowvar=False)
    return mu, cov_matrix

def DecisionTheory(data, X, Y, mu1, mu2, sigma1, sigma2, ground_truth):
    data = projection(X, Y, data)
    loss = 0
    i=0
    for item in data:
        det_cov_1 = np.linalg.det(sigma1)
        inv_cov_1 = np.linalg.inv(sigma1)
        diff = item - mu1
        likehood_1 = (1 / (2 * np.pi * np.sqrt(det_cov_1))) * np.exp(-0.5 * diff.T @ inv_cov_1 @ diff)
        det_cov_2 = np.linalg.det(sigma2)
        inv_cov_2 = np.linalg.inv(sigma2)
        diff = item - mu2
        likehood_2 = (1 / (2 * np.pi * np.sqrt(det_cov_2))) * np.exp(-0.5 * diff.T @ inv_cov_2 @ diff)

        if(likehood_1 >likehood_2):
            if(ground_truth[i] == 1):
                loss +=1
        else:
            if(ground_truth[i] == 0):
                loss +=1
        i+=1
    error_rate = loss/(len(data))
    return error_rate


if __name__ == '__main__':
    #load datasets; Ground Truth: T_shirt = 0, Sneaker = 1
    train_path = os.path.join(os.path.dirname(__file__), 'train_data.mat')
    test_path = os.path.join(os.path.dirname(__file__), 'test_data.mat')

    train_data = scipy.io.loadmat(train_path)
    test_data = scipy.io.loadmat(test_path)
    samples = train_data['x'].reshape(len(train_data['x']), -1)
    samples_test = test_data['x'].reshape(len(test_data['x']), -1)

    #Task 1. Feature normalization (Data conditioning). 
    mean_vector = np.mean(samples, axis=0)
    std_vector =  np.std(samples, axis=0)
    print("Part 1, mean vector = ", mean_vector)
    print("Part 1, std vector = ", std_vector)
    print("==================================================================")
    normalized_samples = normalization(samples, mean_vector, std_vector)
    normalized_samples_test = normalization(samples_test, mean_vector, std_vector)

    #Task 2.  PCA using the training samples. 
    X, Y = PCA(normalized_samples)
    print("First PCA: ", X)
    print("Second PCA:", Y)
    print("==================================================================")
    #Task 3. Dimension reduction using PCA.

    #Train Data
    data_pca_2d = projection(X, Y, normalized_samples)
    T_shirt, Sneaker = view(data_pca_2d, train_data, "Train Dataset")

    #Test Data
    data_pca_2d_test = projection(X, Y, normalized_samples_test)
    view(data_pca_2d_test, test_data, "Test Dataset")

    #Task 4. Density estimation. 
    mu_T_shirt, s_T_shirt = MLE(T_shirt)
    mu_Sneaker, s_Sneaker = MLE(Sneaker)
    print("Estimated mean of T shirt: ",mu_T_shirt)
    print("Estimated cov of T shirt: ",s_T_shirt)
    print("Estimated mean of Sneaker: ",mu_Sneaker)
    print("Estimated cov of Sneaker: ",s_Sneaker)
    print("==================================================================")

    #Task 5. Bayesian Decision Theory for optimal classification. 

    #Train Data
    ground_truth_train = train_data['y'][0]
    error_rate = DecisionTheory(normalized_samples, X, Y, mu_T_shirt, mu_Sneaker, s_T_shirt, s_Sneaker, ground_truth_train)
    print("Error Rate in Training Dataset: ", error_rate)

    #Test Data
    ground_truth_test = test_data['y'][0]
    error_rate = DecisionTheory(normalized_samples_test, X, Y, mu_T_shirt, mu_Sneaker, s_T_shirt, s_Sneaker, ground_truth_test)
    print("Error Rate in Testing Dataset: ", error_rate)
