# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import cv2
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier

# # Load images of healthy and stressed plants
# healthy_images = []
# stressed_images = []
# for file in os.listdir("healthy_plants_folder"):
#     img = cv2.imread("healthy_plants_folder/" + file)
#     healthy_images.append(img)
# for file in os.listdir("stressed_plants_folder"):
#     img = cv2.imread("stressed_plants_folder/" + file)
#     stressed_images.append(img)

# # Combine healthy and stressed plant images
# images = healthy_images + stressed_images

# # # Create labels for the images
# # labels = ["healthy"] * len(healthy_images) + ["stressed"] * len(stressed_images)

# # # Reshape images from 3D to 1D arrays
# # X = []
# # for image in images:
# #     image_reshaped = image.reshape(-1)
# #     X.append(image_reshaped)
# # X = np.array(X)

# # # Encode labels as integers
# # le = LabelEncoder()
# # y = le.fit_transform(labels)

# # # Split the data into training and testing sets
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # # Train a random forest classifier
# # clf = RandomForestClassifier()
# # clf.fit(X_train, y_train)

# # # Make predictions on the test data
# # y_pred = clf.predict(X_test)

# # # Print the classification report
# # print(classification_report(y_test, y_pred))




import cv2
import os
# for img in os.listdir("C:/Users/styli/Downloads/RGB_Augmented/DRstressed"):
#      image=cv2.imread(path+'/'+img)
#      print(image)
#      break
# path="C:/Users/styli/Downloads/RGB_Augmented/DRstressed"
# lis=[ path+'/'+img for img in os.listdir(path)] 
# lis=lis[:50]
# path="C:/Users/styli/Downloads/RGB_Augmented/DRhealthy"
# l=[path+'/'+img for img in os.listdir(path)]
# lis=lis+l[:50]

print(['l' ] *5)
