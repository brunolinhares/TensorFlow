import imutils
from tkinter import ttk
from tkinter import *
from tkinter import font as tkfont
from tkinter import messagebox, PhotoImage, simpledialog
import tkinter as tk
import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np

jan = Tk()
jan.title("Reconhecimento Facial - Painel de Acesso")
jan.geometry("800x480")
jan.configure(background="white")
jan.resizable(width=False, height=False)


def cmd_quit():
    if messagebox.askokcancel("Sair", "Tem certeza?"):
        jan.destroy()


def cap_recog():
    nome = simpledialog.askstring("Nome", "Nome do novo usuário:")
    classificador = cv2.CascadeClassifier("recursos/haarcascade_frontalface_default.xml")
    camera = cv2.VideoCapture(0)
    amostra = 1
    numeroAmostras = 25
    id = nome
    largura, altura = 220, 220
    print("Capturando as faces...")

    while (True):

        conectado, imagem = camera.read()
        imagem = imutils.resize(imagem, width=800, height=480)
        imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        facesDetectadas = classificador.detectMultiScale(imagemCinza,
                                                         scaleFactor=1.5,
                                                         minSize=(150, 150))
        for (x, y, l, a) in facesDetectadas:
            cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
                cv2.imwrite("fotos/pessoas/" + str(id) + "." + str(amostra) + ".jpg", imagemFace)
                print("[foto " + str(amostra) + " capturada com sucesso]")
                amostra += 1

        cv2.imshow("Face", imagem)
        # cv2.waitKey(1)
        if (amostra >= numeroAmostras + 1):
            break

    print("Faces capturadas com sucesso")
    camera.release()
    cv2.destroyAllWindows()


def treinamento_bruno():
    detectorFace = dlib.get_frontal_face_detector()
    detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
    reconhecimentoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")

    indice = {}
    idx = 0
    descritoresFaciais = None

    for arquivo in glob.glob(os.path.join("fotos/pessoas", "*.jpg")):
        imagem = cv2.imread(arquivo)
        facesDetectadas = detectorFace(imagem, 1)
        numeroFacesDetectadas = len(facesDetectadas)
        # print(numeroFacesDetectadas)
        if numeroFacesDetectadas > 1:
            print("Há mais de uma face na imagem {}".format(arquivo))
            messagebox.showerror("Erro", "Há mais de uma face na imagem {}".format(arquivo))
            # exit(0)
        elif numeroFacesDetectadas < 1:
            print("Nenhuma face encontrada no arquivo {}".format(arquivo))
            messagebox.showerror("Erro", "Nenhuma face encontrada no arquivo {}".format(arquivo))
            # exit(0)

        for face in facesDetectadas:
            pontosFaciais = detectorPontos(imagem, face)
            descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)
            print(format(arquivo))
            # print(len(descritorFacial))
            # print(descritorFacial)

            # Converter o descritor de face para o formato dlib para uma lista de 128 caracteristicas
            listaDescritorFacial = [df for df in descritorFacial]
            # print(listaDescrtorFacial)

            npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
            # print(npArrayDescritorFacial)

            npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]

            if descritoresFaciais is None:
                descritoresFaciais = npArrayDescritorFacial
            else:
                descritoresFaciais = np.concatenate((descritoresFaciais, npArrayDescritorFacial), axis=0)

            indice[idx] = arquivo
            idx += 1

        # cv2.imshow("Treinamento", imagem)
        # cv2.waitKey(0)
    # print("Tamanho: {} Formato: {}". format(len(descritoresFaciais), descritoresFaciais.shape))
    np.save("recursos/descritores_rn.npy", descritoresFaciais)
    with open("recursos/indices_rn.pi.ckle", "wb") as f:
        cPickle.dump(indice, f)
        messagebox.showinfo("Treinamento", "Treinamento realizado com sucesso")
    # cv2.destroyAllWindows()


def RF_Bruno2():
    # Load the detector
    detectorFace = dlib.get_frontal_face_detector()

    # Load the predictor
    detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")

    reconhecimentoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")
    indices = np.load("recursos/indices_rn.pickle")
    descritoresFaciais = np.load("recursos/descritores_rn.npy")
    limiar = 0.53

    # read the image
    cap = cv2.VideoCapture(0)

    while True:
        _, imagem = cap.read()
        # Convert image into grayscale

        imagem = imutils.resize(imagem, width=800, height=480)

        gray = cv2.cvtColor(src=imagem, code=cv2.COLOR_BGR2GRAY)

        # Use detector to find landmarks , 1 ou 2
        facesDetectadas = detectorFace(gray)

        for face in facesDetectadas:
            e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
            pontosFaciais = detectorPontos(imagem, face)
            descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)
            listaDescritorFacial = [fd for fd in descritorFacial]
            npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
            npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]

            distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1)
            # print("Distâncias: {}".format(distancias))
            minimo = np.argmin(distancias)
            # print(minimo)
            distanciaMinima = distancias[minimo]
            # print(distanciaMinima)

            if distanciaMinima <= limiar:
                nome = os.path.split(indices[minimo])[1].split(".")[0]
            else:
                nome = " "

            cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 255), 2)

            texto = "{} {:.4f}".format(nome, distanciaMinima)
            cv2.putText(imagem, texto, (d, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255))

        # show the image
        cv2.imshow(winname="Face", mat=imagem)

        # Exit when escape is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the video capture and video write objects
    cap.release()

    # Close all windows
    cv2.destroyAllWindows()


logo = PhotoImage(file="logo.png")
lab = PhotoImage(file="Akaer_Automation.png")

Logolab = Label(jan, image=lab, width=508, height=183, bg="white", relief="flat")
Logolab.place(x=160, y=10)

# LogoLabel = Label(jan, image=logo, width=439, height=78,bg="white", relief="flat")
# LogoLabel.place(x=160, y=550)

new = ttk.Button(jan, text="Cadastro Novo Usuário", width=25, command=cap_recog)
new.place(x=250, y=300)

train = ttk.Button(jan, text="Treinar", width=25, command=treinamento_bruno)
train.place(x=250, y=350)

Recog = ttk.Button(jan, text="Reconhecer", width=25, command=RF_Bruno2)
Recog.place(x=250, y=400)

quit = ttk.Button(jan, text="Sair", width=25, command=cmd_quit)
quit.place(x=600, y=400)

jan.iconphoto(False, tk.PhotoImage(file='icon.ico'))
jan.mainloop()
