import os
import shutil
from tkinter import *
from tkinter import ttk
from PIL import ImageTk,Image
from tkinter import messagebox
from tkinter import filedialog
import tkinter as tk
from ModelBuild import *
from object_detector import *
from GradCam_integ import *
import numpy as np
import keras
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet import preprocess_input

import datetime

import csv
from openpyxl import load_workbook
import re


root=Tk()
root.title('Unique Diamond Sdn. Bhd.')
root.geometry('1500x1000')
root.configure(bg="#fff")
root.resizable(False, False)


def main_menu():
    screen_main = Toplevel(root)
    screen_main.title("GUI Interaction")
    screen_main.geometry("1500x1000")
    screen_main.configure(bg="white")

    option_frame = Frame (screen_main, width=300, height=1000, bg="#c3c3c3", highlightbackground='black', highlightthickness=2)
    option_frame.place(x=0, y=0)


    def about_page():
        about_frame = Frame(screen_main, width=1200, height=1000, bg="white", highlightbackground='black', highlightthickness=2)

        Label(about_frame, text='Data Center Guide',  bg="white", font=('Bold',20)).place(x=100, y=50)
        Label(about_frame, text='1) Enter bad image folder and good image folder accordingly', bg="white", font=('Bold', 20)).place(x=100, y=100)
        Label(about_frame, text='2) Enter the desire folder name', bg="white", font=('Bold', 20)).place(x=100, y=150)
        Label(about_frame, text='3) Observe the data of training dataset', bg="white", font=('Bold', 20)).place(x=100, y=200)

        Label(about_frame, text='Image Processing Guide', bg="white", font=('Bold', 20)).place(x=100, y=500)
        Label(about_frame, text='1) Select the training dataset accordingly', bg="white", font=('Bold', 20)).place(x=100, y=550)
        Label(about_frame, text='2) Capture the image of the product by pressing the button', bg="white", font=('Bold', 20)).place(x=100, y=600)
        Label(about_frame, text='3) Observe the condition of the product with report provided', bg="white", font=('Bold', 20)).place(x=100, y=650)

        about_frame.place(x=300, y=0)

    def dataset_page():
        dataset_frame = Frame(screen_main, width=1200, height=1000, bg="white", highlightbackground='black', highlightthickness=2)

        def training_completion_confirmation(data_train_filename_global):
            global img_tk1
            img_open = Image.open("greentick.png")
            img_resize = img_open.resize((30, 30))
            img_tk1 = ImageTk.PhotoImage(img_resize)
            Label(dataset_frame, image=img_tk1, bg="white").place(x=500, y=390)
            text = data_train_filename_global+".h5"
            Label(dataset_frame, text=text, bg="white", font=('Bold', 15)).place(x=540, y=395)



        def data_tabulate(data_train_filename_global):
            global img_tk
            Label(dataset_frame, text='Accuracy vs Loss', bg="white", font=('Bold', 20)).place(x=230, y=500)
            plot_img_path = "plot" + "/" + data_train_filename_global + ".png"
            img_open = Image.open(plot_img_path)
            img_resize = img_open.resize((400,400))
            img_tk = ImageTk.PhotoImage(img_resize)
            Label(dataset_frame, image=img_tk).place(x=100,y=550)
            train_data_loss_path = "train_data_path"+"/"+data_train_filename_global+"/"+"loss"+"/"+"loss.txt"
            train_data_accuracy_path = "train_data_path"+"/"+data_train_filename_global+"/"+"accuracy"+"/"+"accuracy.txt"
            train_data_improve_loss_path = "train_data_path"+"/"+data_train_filename_global+"/"+"improve loss"+"/"+"loss.txt"
            train_data_improve_accuracy_path = "train_data_path"+"/"+data_train_filename_global+"/"+"improve accuracy"+"/"+"accuracy.txt"

            f1 = open(train_data_loss_path,'r')
            f2 = open(train_data_accuracy_path,'r')
            f3 = open(train_data_improve_loss_path,'r')
            f4 = open(train_data_improve_accuracy_path, 'r')

            train_data_loss = f1.read()
            train_data_accuracy = f2.read()
            train_data_improve_loss = f3.read()
            train_data_improve_accuracy = f4.read()

            Label(dataset_frame, text='Training Data Results', bg="white", font=('Bold', 20)).place(x=670, y=500)
            Label(dataset_frame, text='Mean Loss          :'+train_data_loss, bg="white", font=('Bold', 15)).place(x=650, y=600)
            Label(dataset_frame, text='Mean Accuracy    :' + train_data_accuracy, bg="white", font=('Bold', 15)).place(x=650, y=650)
            Label(dataset_frame, text='Improve Loss        :' + train_data_improve_loss, bg="white", font=('Bold', 15)).place(x=650, y=700)
            Label(dataset_frame, text='Improve Accuracy  :' + train_data_improve_accuracy, bg="white", font=('Bold', 15)).place(x=650, y=750)


        def data_parse():
            global mean_loss
            global mean_accuracy

            main_data_path = r"C:\Users\hojun\PycharmProjects\pythonProject\datapath"
            open_main_data_path = os.listdir(main_data_path)
            main_excel_path = r"C:\Users\hojun\PycharmProjects\pythonProject\data_retrieve"

            if data_train_filename_global in open_main_data_path:
                messagebox.showerror("Error", "File Already Exist. Enter Another Filename")
            else:

                excel_path = main_excel_path+"\\"+data_train_filename_global+".csv"

                with open(excel_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Date & Time", "Object Name", "Condition", "Detect Percentage"])


                data_train_filepath = os.path.join(main_data_path + "\\" + data_train_filename_global)
                os.mkdir(data_train_filepath)

                target_good_filepath = os.path.join(main_data_path + "\\" + data_train_filename_global+ "\\" +"good")
                target_bad_filepath = os.path.join(main_data_path + "\\" + data_train_filename_global+ "\\" +"bad")
                os.mkdir(target_good_filepath)
                os.mkdir(target_bad_filepath)

                origin_good_filepath = ask_directory_good_global
                origin_bad_filepath = ask_directory_bad_global

                open_origin_good_filepath = os.listdir(origin_good_filepath)
                open_origin_bad_filepath = os.listdir(origin_bad_filepath)

                for file_name in open_origin_good_filepath:
                    shutil.copy(origin_good_filepath+"\\"+file_name, target_good_filepath+"\\"+file_name)

                for file_name in open_origin_bad_filepath:
                    shutil.copy(origin_bad_filepath+"\\"+file_name, target_bad_filepath+"\\"+file_name)

                training_data_path = "datapath"+"\\"+data_train_filename_global
                ModelBuild(training_data_path, data_train_filename_global)


        def ask_directory(lb):

            global ask_directory_bad_global
            global ask_directory_good_global

            if lb =="bad":
                ask_directory_bad = filedialog.askdirectory()
                ask_directory_bad_global = ask_directory_bad
                Label(dataset_frame, text=ask_directory_bad_global, bg='#EDFA1E', font=('',15)).place(x=400, y=100)


            if lb =="good":
                ask_directory_good=filedialog.askdirectory()
                ask_directory_good_global = ask_directory_good
                Label(dataset_frame, text=ask_directory_good_global, bg='#EDFA1E', font=('',15)).place(x=400, y=250)


        def data_train_filename():
            global data_train_filename_global
            Data_Train_File_Name1 = Data_Train_File_Name.get()
            data_train_filename_global = Data_Train_File_Name1
            data_parse()
            training_completion_confirmation(data_train_filename_global)
            data_tabulate(data_train_filename_global)




        Label(dataset_frame, text='Enter Bad Folder Directory',  bg="white", font=('Bold',15)).place(x=100, y=50)
        Bad_Button=Button(dataset_frame, width=34, pady=18, text='Bad Directory', bg='grey', fg='white',border=0, command=lambda :ask_directory("bad"))
        Bad_Button.place(x=100, y=90)

        Label(dataset_frame, text='Enter Good Folder Directory', bg="white", font=('Bold', 15)).place(x=100, y=200)
        Good_Button=Button(dataset_frame, width=34, pady=18, text='Good Directory', bg='grey', fg='white',border=0, command=lambda :ask_directory("good"))
        Good_Button.place(x=100, y=240)

        Label(dataset_frame, text='Enter File Name', bg="white", font=('Bold', 15)).place(x=100, y=350)
        Data_Train_File_Name = Entry(dataset_frame, bg='light grey', font=('Bold', 15))
        Data_Train_File_Name.place(x=100, y=390, width=200,height=40)

        Parse_button = Button(dataset_frame, width=10, pady=5, text='Parse', bg='#EC1305', fg='white',border=0, font=('Bold',15), command=lambda :data_train_filename())
        Parse_button.place(x=350, y=390)

        dataset_frame.place(x=300, y=0)
        dataset_frame.mainloop()

    def image_processing_page():
        global img
        global cam_on, cap
        global capture
        global model_load
        global model_file

        image_processing_frame = Frame(screen_main, width=1200, height=1000, bg="white", highlightbackground='black', highlightthickness=2)
        camera_frame = Frame(image_processing_frame, width = 850, height=400,bg='black', highlightbackground='grey', highlightthickness=3)
        tabulate_frame = Frame(image_processing_frame, width=500,height=300, bg='white', highlightbackground='light grey', highlightthickness=3)
        Label(image_processing_frame, text='Camera Frame', bg="white", font=('Bold', 15)).place(x=100, y=50)


        cam_on = False
        cap = None
        detector = HomogeneousBgDetector()
        framing = Label(camera_frame, width=850, height=400)




        def show_frame():
            global capture
            if cam_on:
                global capture
                ret, img = cap.read()
                contours = detector.detect_objects(img)

                for cnt in contours:
                    # Get rect
                    rect = cv2.minAreaRect(cnt)
                    (x, y), (w, h), angle = rect
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv2.polylines(img, [box], True, (255, 0, 0), 2)

                opencv_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
                captured_image = Image.fromarray(opencv_image)
                photo_image = ImageTk.PhotoImage(image=captured_image)
                framing.photo_image = photo_image
                framing.config(image=photo_image)
                framing.pack()
                framing.after(10, show_frame)

                if capture:
                    capture_path = r"C:\Users\hojun\PycharmProjects\pythonProject\capture_path"
                    capture_image = model_file[:-3]
                    capture_image_path = capture_path+"\\"+capture_image

                    if capture_image not in os.listdir(capture_path):
                        os.mkdir(capture_image_path)
                        #print(capture_image)
                        print("capturing")
                        capture_image_path_save = capture_image_path+"\\"+capture_image+"_1.png"
                        capture_name = capture_image+"_1"
                        test, img1 = cap.read()
                        cv2.imwrite(capture_image_path_save, img1)
                        capture = False
                        tabulate_result(capture_image_path_save,capture_name, capture_image)
                    else:
                        print("is in")
                        #print(capture_image)
                        print("capturing")
                        listing = os.listdir(capture_image_path)
                        last_of_listing = listing[-1]
                        #print(last_of_listing)
                        obtain_integer = int((re.sub(".*_",'', last_of_listing)).strip(".png"))
                        integer = obtain_integer+1
                        capture_image_path_save = capture_image_path+"\\"+capture_image+"_"+str(integer)+".png"
                        capture_name = capture_image+"_"+str(integer)
                        print(capture_image_path_save)
                        test, img1 = cap.read()
                        cv2.imwrite(capture_image_path_save, img1)
                        capture = False
                        tabulate_result(capture_image_path_save, capture_name, capture_image)




        def start_video():
            global cam_on, cap, capture
            print("in start video")
            stop_video()
            cam_on = True
            capture = False
            cap = cv2.VideoCapture(1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1100)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
            show_frame()


        def stop_video():
            global cam_on
            print("in stop video")
            cam_on = False

            if cap:
                cap.release()
                img=Image.open("black_background.jpg")
                img_resize=img.resize((850,400))
                photo_image = ImageTk.PhotoImage(image=img_resize)
                framing.photo_image = photo_image
                framing.config(image=photo_image)
                framing.pack()


        def tabulate_result(capture_image_path_save, capture_name, capture_image):
            global gradpltimg_tk
            global gradimg_tk
            #image = cv2.imread("successhaha.png")
            #cv2.imshow("Image", image)
            #print(capture_image_path_save)
            image_path = capture_image_path_save
            detect_image = load_img(image_path, target_size=(256,256))

            i=img_to_array(detect_image)/255
            input_arr = np.array([i])
            input_arr.shape

            pred = np.argmax(model_load.predict(input_arr))  # print index value
            pred1 = np.max(model_load.predict(input_arr))  # print sum value
            prediction_percentage = pred1 * 100
            prediction_percentage = "{:.4f}".format(prediction_percentage)
            prediction_percentage = str(prediction_percentage) + "%"
            GradCam_integ(model_load, image_path)
            gradpltimg = Image.open("heatmapplt.jpg")
            gradimg = Image.open("heatmap.jpg")
            gradpltimg_resize = gradpltimg.resize((300,200))
            gradimg_resize = gradimg.resize((250,150))
            gradpltimg_tk = ImageTk.PhotoImage(gradpltimg_resize)
            gradimg_tk = ImageTk.PhotoImage(gradimg_resize)

            if pred == 0:
                time_now = datetime.datetime.now()
                time_now = time_now.strftime("%Y-%m-%d %H:%M:%S")
                excel_path = r"C:\Users\hojun\PycharmProjects\pythonProject\data_retrieve" + "\\"+capture_image+".csv"

                with open(excel_path, 'a', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([time_now, capture_name, "bad", prediction_percentage])


                Label(tabulate_frame, text='Name Of Test                  : '+capture_name, bg="white", font=('Bold', 15)).place(x=10, y=10)
                Label(tabulate_frame, text='Status Of Test                 : ', bg="white", font=('Bold', 15)).place(x=10, y=60)
                Label(tabulate_frame, text='FAIL', bg = "red", font=('Bold', 15)).place(x=260, y=60)
                Label(tabulate_frame, text='Test Prediction Percentage: ', bg="white", font=('Bold', 15)).place(x=10, y=120)
                Label(tabulate_frame, text=prediction_percentage, bg="white", font=('Bold', 15)).place(x=260, y=120)
                Label(image_processing_frame, text="Gradcam Plot", bg="white", font=('Bold', 13)).place(x=790, y=550)
                Label(image_processing_frame, image=gradpltimg_tk, bg = "white").place(x=700, y=570)
                Label(image_processing_frame, text="Gradcam Image", bg="white", font=('Bold', 13)).place(x=780, y=770)
                Label(image_processing_frame, image=gradimg_tk, bg="white").place(x=740, y=800)

            else:
                time_now = datetime.datetime.now()
                time_now = time_now.strftime("%Y-%m-%d %H:%M:%S")
                excel_path = r"C:\Users\hojun\PycharmProjects\pythonProject\data_retrieve" + "\\" + capture_image + ".csv"

                with open(excel_path, 'a', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([time_now, capture_name, "bad", prediction_percentage])
                Label(tabulate_frame, text='Name Of Test                  : ' + capture_name, bg="white",font=('Bold', 15)).place(x=10, y=10)
                Label(tabulate_frame, text='Status Of Test                 : ', bg="white", font=('Bold', 15)).place(x=10, y=60)
                Label(tabulate_frame, text='PASS', bg="green", font=('Bold', 15)).place(x=260, y=60)
                Label(tabulate_frame, text='Test Prediction Percentage: ', bg="white", font=('Bold', 15)).place(x=10,y=120)
                Label(tabulate_frame, text=prediction_percentage, bg="white", font=('Bold', 15)).place(x=260, y=120)
                Label(image_processing_frame, text="Gradcam Plot", bg="white", font=('Bold', 13)).place(x=790, y=550)
                Label(image_processing_frame, image=gradpltimg_tk, bg="white").place(x=700, y=570)
                Label(image_processing_frame, text="Gradcam Image", bg="white", font=('Bold', 13)).place(x=780, y=770)
                Label(image_processing_frame, image=gradimg_tk, bg="white").place(x=740, y=800)




        def capture_image():
            global capture
            capture = True

        Label(image_processing_frame, text='Test Result', bg="white", font=('Bold', 15)).place(x=100, y=550)
        Label(image_processing_frame, text='Load Model', bg="white", font=('Bold', 10)).place(x=530, y=30)
        train_model_path = r"C:\Users\hojun\PycharmProjects\pythonProject\model"
        list_model_path = os.listdir(train_model_path)

        def pickModel(e):
            global model_load
            global model_file
            model_file = drop_down_menu.get()
            model_full_path_file = train_model_path+"\\"+model_file
            model_load = load_model(model_full_path_file)
            #print(model_full_path_file)


        drop_down_menu = ttk.Combobox(image_processing_frame, value=list_model_path)
        drop_down_menu.place(x=500, y=50)
        drop_down_menu.bind("<<ComboboxSelected>>", pickModel)

        Start_Camera_Button = Button(image_processing_frame, width=10, pady=5, text='Start', bg='#57a1f8', fg='white',border=0, font=('Bold', 15), command=start_video)
        Start_Camera_Button.place(x=1000, y=100)
        Stop_Camera_Button = Button(image_processing_frame, width=10, pady=5, text='Stop', bg='#EC1305', fg='white',border=0, font=('Bold', 15), command=stop_video)
        Stop_Camera_Button.place(x=1000, y=150)
        Capture_Camera_Button = Button(image_processing_frame, width=10, pady=5, text='Capture', bg='#88F0A4', fg='white',border=0, font=('Bold', 15), command=capture_image)
        Capture_Camera_Button.place(x=1000, y=250)
        image_processing_frame.place(x=300, y=0)
        camera_frame.place(x=100, y=100)
        tabulate_frame.place(x=100, y=600)

    def data_retrieve_page():
        global excel_frame,tv1, treescrolly, treescrollx
        global data_retrieve_file
        data_retrieve_frame = Frame(screen_main, width=1200, height=1000, bg="white", highlightbackground='black',highlightthickness=2)
        excel_frame = tk.LabelFrame(data_retrieve_frame, text="Excel Data")


        #data_retrieve_button = Button(data_retrieve_frame, width=20, pady=5, text = "Data Retrieve", bg="grey",border=0, font=('Bold', 15))
        #data_retrieve_button.place(x=500, y=500)

        #Treeview Widget
        tv1 = ttk.Treeview(excel_frame)
        tv1.place(relheight=1, relwidth=1)  # set the height and width of the widget to 100% of its container (frame1).

        treescrolly = tk.Scrollbar(excel_frame, orient="vertical",command=tv1.yview)  # command means update the yaxis view of the widget
        treescrollx = tk.Scrollbar(excel_frame, orient="horizontal",command=tv1.xview)  # command means update the xaxis view of the widget
        tv1.configure(xscrollcommand=treescrollx.set,yscrollcommand=treescrolly.set)  # assign the scrollbars to the Treeview Widget
        treescrollx.pack(side="bottom", fill="x")  # make the scrollbar fill the x axis of the Treeview widget
        treescrolly.pack(side="right", fill="y")  # make the scrollbar fill the y axis of the Treeview widget

        data_retrieve_path = r"C:\Users\hojun\PycharmProjects\pythonProject\data_retrieve"
        list_data_retrieve_path = os.listdir(data_retrieve_path)

        def pickModel(e):
            global data_retrieve_file
            data_retrieve_file = drop_down_menu.get()

        Label(data_retrieve_frame, text="Data File", bg="white", font=('Bold', 15)).place(x=100, y=50)
        drop_down_menu = ttk.Combobox(data_retrieve_frame, value=list_data_retrieve_path)
        drop_down_menu.place(x=100, y=100, width=200)
        drop_down_menu.bind("<<ComboboxSelected>>", pickModel)

        def clear_data():
            tv1.delete(*tv1.get_children())
            return None

        def Load_File():
            file_path = data_retrieve_path+"\\"+data_retrieve_file

            try:
                if file_path[-4:]==".csv":
                    df = pd.read_csv(file_path)
            except FileNotFoundError:
                messagebox.showerror("Information", f"No such file as {file_path}")
            clear_data()

            tv1["column"] = list(df.columns)
            tv1["show"] = "headings"
            for column in tv1["columns"]:
                tv1.heading(column, text=column)  # let the column heading = column name

            df_rows = df.to_numpy().tolist()  # turns the dataframe into a list of lists
            for row in df_rows:
                tv1.insert("", "end",
                           values=row)  # inserts each list into the treeview. For parameters see https://docs.python.org/3/library/tkinter.ttk.html#tkinter.ttk.Treeview.insert
            return None


        Load_File_Button = Button(data_retrieve_frame, width=10, pady=5, text="Load File", bg='grey', fg='white',border=0, font=('Bold', 15),command=Load_File)
        Load_File_Button.place(x=380, y=80)

        excel_frame.place(height = 700, width=1000, x=100, y=200)
        data_retrieve_frame.place(x=300, y=0)


    def hide_indicators():
        about_indicate.config(bg='#c3c3c3')
        data_center_indicate.config(bg='#c3c3c3')
        image_processing_indicate.config(bg='#c3c3c3')
        data_retrieve_indicate.config(bg='#c3c3c3')

    def indicate(lb, page):
        hide_indicators()
        lb.config(bg='#000000')
        page()

    about_button = Button(option_frame, text="About", font=("Bold",20), fg="#000000", bd=0, bg="#c3c3c3", command=lambda :indicate(about_indicate, about_page))
    about_button.place(x=100, y=90)

    about_indicate = Label(option_frame, text="", bg="#c3c3c3")
    about_indicate.place(x=3, y=80, width=10, height=70)


    data_center_button = Button(option_frame, text="Data Center", font=("Bold",20), fg="#000000", bd=0, bg="#c3c3c3", command=lambda :indicate(data_center_indicate, dataset_page))
    data_center_button.place(x=60, y=270)

    data_center_indicate = Label(option_frame, text="", bg="#c3c3c3")
    data_center_indicate.place(x=3, y=260, width=10, height=70)

    image_processing_button = Button(option_frame, text="Image Processing", font=("Bold",20), fg="#000000", bd=0, bg="#c3c3c3", command=lambda :indicate(image_processing_indicate, image_processing_page))
    image_processing_button.place(x=30, y=470)

    image_processing_indicate = Label(option_frame, text="", bg="#c3c3c3")
    image_processing_indicate.place(x=3, y=460, width=10, height=70)

    data_retrieve_button = Button(option_frame, text="Data Retrieve", font=("Bold",20), fg="#000000", bd=0, bg="#c3c3c3", command=lambda :indicate(data_retrieve_indicate, data_retrieve_page))
    data_retrieve_button.place(x=60, y=700)

    data_retrieve_indicate = Label(option_frame, text="", bg="#c3c3c3")
    data_retrieve_indicate.place(x=3, y=690, width=10, height=70)


def signin():
    username1 = user.get()
    password1 = password.get()

    username_file = (username1+".txt")
    file = "SignUp"
    list_of_files = os.listdir(file)

    if username_file in list_of_files:
        password_file = rf'{file}\{username_file}'
        with open(password_file, 'r') as f:
            password_check = f.read()
            if password1 == password_check:
                messagebox.showinfo("Login", "Login Successful")
                main_menu()

            else:
                messagebox.showerror("Error", "Wrong Password")

    else:
        messagebox.showerror("Error", "Wrong ID")


def signup():
    screen2 = Toplevel(root)
    screen2.title("Sign Up")
    screen2.geometry('1500x1000')
    screen2.configure(bg="#c3c3c3",border=10)


    def signup_check():
        ID = CVT_ID.get()
        EMAIL = CVT_EMAIL.get()
        password = Password.get()

        ID_confirm=(ID+".txt")
        path = "ID"
        open_file = os.listdir(path)


        if ID_confirm in open_file:
            file = rf'{path}\{ID_confirm}'
            with open(file, 'r') as f:
                email = f.read()
                if email == EMAIL:
                    path1 = "SignUp"
                    sign_up_file = rf'{path1}\{ID_confirm}'
                    file1 = open(sign_up_file,'w')
                    file1.write(password)
                    messagebox.showinfo("SignUp","SignUp Sucessfully")
                    screen2.destroy()

                else:
                    messagebox.showerror("Error", "Wrong Email")
                    screen2.destroy()

        else:
            messagebox.showerror("Error", "Wrong ID")
            screen2.destroy()

    Label(screen2, image=img1).place(x=50, y=70)

    frame1=Frame(screen2, width=500, height=800, bg='white')
    frame1.place(x=830, y=80)

    Label(screen2, text="SIGN UP PAGE", fg="#0d0c0c", bg="#c3c3c3", font=('Microsoft YaHei UI Light', 40, 'bold')).place(x=180, y=80)
    Label(screen2, text="1) Only Computer Vision Team (CVT) can sign up", fg="#0d0c0c", bg="#c3c3c3", font=('Microsoft YaHei UI Light', 23, 'bold')).place(x=50, y=300) #yellow background
    Label(screen2, text="2) Please enter valid CVT ID", fg="#0d0c0c", bg="#c3c3c3", font=('Microsoft YaHei UI Light', 23, 'bold')).place(x=50, y=400)
    Label(screen2, text="3) Please enter valid CVT Email", fg="#0d0c0c", bg="#c3c3c3", font=('Microsoft YaHei UI Light', 23, 'bold')).place(x=50, y=500)

    Label(frame1, text="Sign Up", fg="#0d0c0c", bg="white",font=('Microsoft YaHei UI Light', 23, 'bold')).place(x=170, y=50)
    Label(frame1, text="CVT ID        :", fg="#0d0c0c", bg="white",font=('Microsoft YaHei UI Light', 15, 'bold')).place(x=30, y=150)
    CVT_ID = Entry(frame1, width=30)
    CVT_ID.place(x=170, y=160)
    Label(frame1, text="CVT EMAIL  :", fg="#0d0c0c", bg="white", font=('Microsoft YaHei UI Light', 15, 'bold')).place(x=30, y=250)
    CVT_EMAIL = Entry(frame1, width=30)
    CVT_EMAIL.place(x=170, y=260)
    Label(frame1, text="PASSWORD :", fg="#0d0c0c", bg="white", font=('Microsoft YaHei UI Light', 15, 'bold')).place(x=30, y=350)
    Password = Entry(frame1, width=30)
    Password.place(x=170, y=360)

    SignUp = Button(frame1, width=23, pady=20, text='Sign Up', bg='#e0281b', fg='white',border=0, command=signup_check)
    SignUp.place(x=170, y=480)

image1 = Image.open("diamondlogo.png")
imgs1 = image1.resize((100,100))
img1 = ImageTk.PhotoImage(imgs1)

image = Image.open('Diamond.png')
imgs = image.resize((1300,450))
img = ImageTk.PhotoImage(imgs)
Label(root,image=img, bg='white').place(x=100,y=100)

frame=Frame(root,width=900,height=400,bg='white')
frame.place(x=350, y=650)

heading = Label(frame, text='Quality Inspection Using Image Processing',fg='#111214',bg='white', font=('Microsoft YaHei UI Light', 23, 'bold'))
heading.place(x=100,y=5)


def on_enter(e):
    user.delete(0,'end')

def on_leave(e):
    name=user.get()
    if name=='':
        user.insert(0,'Username')

user = Entry(frame, width=25,fg='black', border=0, bg='white', font=('Microsoft YaHei UI Light', 11))
user.place(x=350,y=100)
user.insert(0,'Username')
user.bind('<FocusIn>', on_enter)
user.bind('<FocusOut>', on_leave)

Frame(frame, width=295, height=2, bg='black').place(x=250, y=130)

def on_enter(e):
    password.delete(0,'end')

def on_leave(e):
    name=password.get()
    if name=='':
        password.insert(0,'Password')

password=Entry(frame, width=25,fg='black', border=0, bg='white', font=('Microsoft YaHei UI Light', 11))
password.place(x=350,y=170)
password.insert(0,'Password')
password.bind('<FocusIn>',on_enter)
password.bind('<FocusOut>',on_leave)


Frame(frame, width=295, height=2, bg='black').place(x=250, y=200)

Signin = Button(frame, width=23, pady=7, text='Sign In', bg='#57a1f8', fg='white',border=0, command=signin)
Signin.place(x=400,y=250)

Signup = Button(frame, width=23, pady=7, text='Sign Up', bg='#e0281b', fg='white',border=0, command=signup)
Signup.place(x=200,y=250)


root.mainloop()