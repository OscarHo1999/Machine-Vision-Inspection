# Machine-Vision-Inspection
Machine Vision Inspection Program
A machine vision inspection program is a computer program designed to analyze images or video streams and identify defects, anomalies, or patterns of interest. It uses various techniques such as image processing, computer vision, and machine learning algorithms to extract information from visual data.

The main goal of a machine vision inspection program is to automate the inspection process and improve the accuracy and efficiency of quality control in manufacturing or other industrial applications. It can be used to inspect products, materials, or surfaces for defects, measure dimensions, detect contaminants, and ensure compliance with quality standards.

In this program, user can train dataset on different product and perform real-time detection to monitor good and defect product. This program consists of 4 part which is about section, dataset training section, real-time monitoring section, and data-retrieve section. To ease the visualization of user, the program was developed with Graphical User Interface using Tkinter library. 

### Login Registration
When the program first started, it will direct user to the Login and Registration Page. 
(1) User enter their username and password accordingly
(2) If either username or password does not match the data system, it will flag error 
(2) If username and password match the data system it will proceed to main page

![Image](https://user-images.githubusercontent.com/67437888/235358112-bc604d63-1f48-4bc4-a9c3-ec790e1cd3b5.JPG)

User can perform registration at the registration page provided it's email and ID exists in the data system 
(1) User can enter their ID, email, password to create an account. 
(2) if email or ID does not match the data system, it will not create an account for the user and will flag error. 
(3) if email and ID match the data system, it will create an account for the user


![Image](https://user-images.githubusercontent.com/67437888/235358444-cca598fb-e97a-4ce9-9098-ae169b75c075.JPG)

### About 
In about page, it will explain the usage of the program accordingly


![Image](https://user-images.githubusercontent.com/67437888/235358616-55f95bb0-1bcf-44e3-9eba-62fa043ed939.JPG)

### Model Training
In data center page, it is the dataset training section where user have to provide good and defect image as this is a supervised machine learning algorithm. Convolutional Neural Network (CNN) deep learning algorithm was used as the algorithm for the trained data. 


![Image](https://user-images.githubusercontent.com/67437888/235358893-01e2bfc3-fdad-4345-af79-6444d6cb484a.JPG)

### Real-Time Monitoring
In image processing page, user will have to load their desire model accordingly. After the model was loaded, START button needed to be pressed to start the camera action. Inside the camera, object tracking was used by drawing boundary box around the object. As my face was not the object that was desired to be tested, it will return 100% predicted defect image. 


![Image](https://user-images.githubusercontent.com/67437888/235359655-6a44be3a-d932-4c2f-911e-a5f2e1fd0705.JPG)

If the object was a good product, it will return a good product statement 


![Image](https://user-images.githubusercontent.com/67437888/235359878-f94fbaf2-ba22-4395-a692-5e33866ca8c7.jpeg)


### Data Retrieve
In data retrieve page, user can collect the data of the respective model by just clicking to the desired csv file and click the button LOAD



![Image](https://user-images.githubusercontent.com/67437888/235360125-e120b900-98b4-4f1c-97e6-02dfd25febee.png)
