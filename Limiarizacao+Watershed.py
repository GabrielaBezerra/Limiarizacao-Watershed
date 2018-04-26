from numpy import *
from cv2 import *

img = imread('imgs/coins.png')
imshow('coins.png',img)

# Limiarizacao
gray = cvtColor(img,COLOR_BGR2GRAY)
ret, thresh = threshold(gray,0,255,THRESH_BINARY_INV+THRESH_OTSU)
imshow('Apos limizarizacao',thresh)


# Remocao de ruidos
kernel = ones((3,3),uint8)
opening = morphologyEx(thresh,MORPH_OPEN,kernel, iterations = 2)

# Extracao de background
sure_bg = dilate(opening,kernel,iterations=3)

# Extracao de foreground
dist_transform = distanceTransform(opening,DIST_L2,5)
ret, sure_fg = threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Encontrando regiao desconhecida
sure_fg = uint8(sure_fg)
unknown = subtract(sure_bg,sure_fg)

# Desenhando marcadores
ret, markers = connectedComponents(sure_fg)

# Adicionando 1 para todos os rotulos, diferenciando o background, que tera valor 1
markers = markers+1

# Marcando a regiao desconhecida com o valor 0
markers[unknown==255] = 0


# Aplicando watershed
markers = watershed(img,markers)
img[markers == -1] = [255,0,0]

imshow("Deteccao de bordas com o Watershed",img)
waitKey(0)

