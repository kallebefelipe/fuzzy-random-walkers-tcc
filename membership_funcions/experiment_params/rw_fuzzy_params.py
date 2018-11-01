
import numpy as np
from constants import beta, tol, mode, images
from math import sqrt
from skimage import io
from skimage import segmentation
from skimage.segmentation import random_walker
import scipy.io as sio
import csv
import math
import scipy.io
import scipy.misc
from math import exp


def run(function, params):
    soma = {}
    media = {}
    medias = {}

    numero_marcacao = []
    seeds = 6
    with open('../../data/output/rw_fuzzy_params/'+function+'/resultado.csv', "w") as \
            resultado:
        output = csv.writer(resultado, quoting=csv.QUOTE_ALL)

        cabecalho_1 = []
        cabecalho_1.append("")
        for a in beta:
            for b in tol:
                for _c in mode:
                    titulo = str(' beta='+str(a)+' tol='+str(b) +
                                 ' mode='+str(_c)+' ')
                    cabecalho_1.append(titulo)
                    cabecalho_1.append('')
                    cabecalho_1.append('')
                    cabecalho_1.append('')
                    cabecalho_1.append('')
                    cabecalho_1.append('')
                    cabecalho_1.append('')
                    cabecalho_1.append('')

        output.writerow(cabecalho_1)

        cabecalho_1 = []
        cabecalho_1.append("")
        for a in params['params']:
            titulo = str(params['name']+' ='+str(a))
            cabecalho_1.append(titulo)
            cabecalho_1.append('')
            cabecalho_1.append('')
            cabecalho_1.append('')
            cabecalho_1.append('')
            cabecalho_1.append('')
            cabecalho_1.append('')
            cabecalho_1.append('')

        output.writerow(cabecalho_1)

        cabecalho_2 = []
        cabecalho_2.append('')
        cont = 0
        for a in params['params']:
            cabecalho_2.append('TP')
            cabecalho_2.append('TN')
            cabecalho_2.append('FP')
            cabecalho_2.append('FN')
            cabecalho_2.append('Sensibility')
            cabecalho_2.append('Especificy')
            cabecalho_2.append('Jaccard')
            cabecalho_2.append('TFP')
            cont += 8
        output.writerow(cabecalho_2)

        for param in params['params']:
            media[param] = ['media']
            medias[param] = []
            soma[param] = [0] * cont

        for count, i in enumerate(images):
            linha = []
            linha.append(i)
            for param in params['params']:
                pos = 0

                numbMarkx = 0
                numbMarky = 0
                posX = [0] * 6
                posY = [0] * 6
                posX_origem = [0] * 6
                posY_origem = [0] * 6
                image = io.imread('../../seeds/imagens/'+i+'.bmp')
                markers = np.zeros((image.shape[0], image.shape[1]))

                mat_contents = sio.loadmat('../../seeds/labels_roi/'+i+'.mat')
                oct_cells = mat_contents['labels']
                val = oct_cells[0, 1]
                markers = np.zeros((image.shape[0], image.shape[1]))

                for x in range(0, image.shape[0]):
                    for y in range(0, image.shape[1]):
                        val1 = oct_cells[x, y]

                        if val1 == 1:
                            posX[numbMarkx] = x         # atribuicao no vetor de x's
                            posY[numbMarky] = y         # atribuicao no vetor de y's
                            numbMarkx = numbMarkx + 1
                            numbMarky = numbMarky + 1
                colb = 1
                posX = list(map(lambda x: x/float(image.shape[0]), posX))
                posY = list(map(lambda x: x/float(image.shape[1]), posY))

                copia = io.imread('../../seeds/imagens/'+i+'.bmp')
                imageShape = np.ones((image.shape[0], image.shape[1]))

                xm = 0
                ym = 0
                ax = 1
                ay = 1
                desviox = 0
                desvioy = 0
                constantex = 1
                constantey = 1

                for x in range(0, numbMarkx):
                    xm = xm + posX[x]
                xm = xm / numbMarkx

                for y in range(0, numbMarky):
                    ym = ym + posY[y]
                ym = ym / numbMarky

                somadorx = 0

                for x in range(0, numbMarkx):
                    somadorx = somadorx + (math.pow((posX[x] - xm), 2))
                desviox = sqrt(somadorx / float(numbMarkx-1))
                somadory = 0

                for y in range(0, numbMarky):
                    somadory = somadory + (math.pow((posY[y] - ym), 2))

                desvioy = sqrt(somadory / float(numbMarky-1))

                if function == 'fuzzy':

                    for x in range(0, image.shape[0]):
                        for y in range(0, image.shape[1]):
                            x_n = x/float(image.shape[0])
                            y_n = y/float(image.shape[1])
                            imageShape[x, y] = \
                                math.exp(((-1)*math.pow((x_n-xm), 2)) / float(2 * constantex * desviox) + ((-1) * math.pow((y_n-ym), 2))/float(2*constantey*desvioy))
                elif function == 'triangular':
                    beta_func = param
                    ax = xm - (desviox*beta_func)
                    cx = xm + (desviox*beta_func)
                    ay = ym - (desvioy*beta_func)
                    cy = ym + (desvioy*beta_func)

                    for x in range(0, image.shape[0]):
                        for y in range(0, image.shape[1]):
                            x_n = x/float(image.shape[0])
                            y_n = y/float(image.shape[1])

                            ma = max(min((x_n-ax)/(xm-ax), (cx-x_n)/(cx-xm)), 0)
                            mb = max(min((y_n-ay)/(ym-ay), (cy-y_n)/(cy-y_n)), 0)
                            imageShape[x, y] = min(ma, mb)
                elif function == 'trapezoidal':
                    beta_func = param
                    ax = xm - (desviox*beta_func)
                    bx = xm - ((desviox*beta_func)/2)
                    cx = xm + ((desviox*beta_func)/2)
                    dx = xm + (desviox*beta_func)

                    ay = ym - (desvioy*beta_func)
                    bb = ym - ((desvioy*beta_func)/2)
                    bc = ym + ((desvioy*beta_func)/2)
                    bd = ym + (desvioy*beta_func)

                    for x in range(0, image.shape[0]):
                        for y in range(0, image.shape[1]):
                            x_n = x/float(image.shape[0])
                            y_n = y/float(image.shape[1])
                            ma = max(min([(x_n-ax)/(bx-ax), 1, (dx-x_n)/(dx-cx)]), 0)
                            mb = max(min([(y_n-ay)/(bb-ay), 1, (bd-y_n)/(bd-bc)]), 0)
                            imageShape[x, y] = min(ma, mb)
                elif function == 'gaussian':
                    alfa = param
                    for x in range(0, image.shape[0]):
                        for y in range(0, image.shape[1]):
                            x_n = x/float(image.shape[0])
                            y_n = y/float(image.shape[1])
                            imageShape[x, y] = exp(-((x_n-xm) ** 2)/(2*alfa*(
                                desviox ** 2))) * exp(-((y_n-ym) ** 2)/(2*alfa*(desvioy ** 2)))
                elif function == 'bell':
                    b = param
                    for x in range(0, image.shape[0]):
                        for y in range(0, image.shape[1]):
                            x_n = x/float(image.shape[0])
                            y_n = y/float(image.shape[1])
                            ma = 1/((abs((x_n/(2*desviox))-(xm/(2*desviox)))**(2*b))+1)
                            mb = 1/((abs((y_n/(2*desvioy))-(ym/(2*desvioy)))**(2*b))+1)
                            imageShape[x, y] = ma*mb
                elif function == 'mult_gaussian':

                    for x in range(0, image.shape[0]):
                        for y in range(0, image.shape[1]):
                            x_n = x/float(image.shape[0])
                            y_n = y/float(image.shape[1])

                            p_alfa = param

                            if max(desvioy, desviox) >= 27:
                                desv = min(desvioy, desviox)
                            else:
                                desv = max(desvioy, desviox)

                            desv = max(desvioy, desviox)

                            probObj_p1 = exp(-((x_n-posX[0])**2)/(2*p_alfa*(desv**2))) * exp(-((y_n-posY[0])**2)/(2*p_alfa*(desv**2)))
                            probObj_p2 = exp(-((x_n-posX[1])**2)/(2*p_alfa*(desv**2))) * exp(-((y_n-posY[1])**2)/(2*p_alfa*(desv**2)))
                            probObj_p3 = exp(-((x_n-posX[2])**2)/(2*p_alfa*(desv**2))) * exp(-((y_n-posY[2])**2)/(2*p_alfa*(desv**2)))
                            probObj_p4 = exp(-((x_n-posX[3])**2)/(2*p_alfa*(desv**2))) * exp(-((y_n-posY[3])**2)/(2*p_alfa*(desv**2)))
                            probObj_p5 = exp(-((x_n-posX[4])**2)/(2*p_alfa*(desv**2))) * exp(-((y_n-posY[4])**2)/(2*p_alfa*(desv**2)))
                            probObj_p6 = exp(-((x_n-posX[5])**2)/(2*p_alfa*(desv**2))) * exp(-((y_n-posY[5])**2)/(2*p_alfa*(desv**2)))

                            alfacenter = 1

                            probObjcenter = exp(-((x_n-desviox)**2)/(2*alfacenter*(desviox**2))) * exp(-((y_n-desvioy)**2)/(2*alfacenter*(desvioy**2)))
                            probObjcenter = 0
                            imageShape[x, y] = max([probObj_p1, probObj_p2, probObj_p3, probObj_p4, probObj_p5, probObj_p6, probObjcenter])
                elif function == 'mult_fuzzy':

                    for x in range(0, image.shape[0]):
                        for y in range(0, image.shape[1]):
                            x_n = x/float(image.shape[0])
                            y_n = y/float(image.shape[1])

                            p_alfa = param

                            if max(desvioy, desviox) >= 27:
                                desv = min(desvioy, desviox)
                            else:
                                desv = max(desvioy, desviox)

                            desv = max(desvioy, desviox)

                            probObj_p1 = exp(-((x_n-posX[0]))/(2*p_alfa*(desv))) * exp(-((y_n-posY[0]))/(2*p_alfa*(desv)))
                            probObj_p2 = exp(-((x_n-posX[1]))/(2*p_alfa*(desv))) * exp(-((y_n-posY[1]))/(2*p_alfa*(desv)))
                            probObj_p3 = exp(-((x_n-posX[2]))/(2*p_alfa*(desv))) * exp(-((y_n-posY[2]))/(2*p_alfa*(desv)))
                            probObj_p4 = exp(-((x_n-posX[3]))/(2*p_alfa*(desv))) * exp(-((y_n-posY[3]))/(2*p_alfa*(desv)))
                            probObj_p5 = exp(-((x_n-posX[4]))/(2*p_alfa*(desv))) * exp(-((y_n-posY[4]))/(2*p_alfa*(desv)))
                            probObj_p6 = exp(-((x_n-posX[5]))/(2*p_alfa*(desv))) * exp(-((y_n-posY[5]))/(2*p_alfa*(desv)))

                            alfacenter = 1

                            probObjcenter = exp(-((x_n-desviox)**2)/(2*alfacenter*(desviox**2))) * exp(-((y_n-desvioy)**2)/(2*alfacenter*(desvioy**2)))
                            probObjcenter = 0
                            imageShape[x, y] = max([probObj_p1, probObj_p2, probObj_p3, probObj_p4, probObj_p5, probObj_p6, probObjcenter])

                for x in range(0, image.shape[0]):
                    for y in range(0, image.shape[1]):
                        if(imageShape[x, y] > 0.5):
                            copia[x, y] = 1
                        else:
                            copia[x, y] = 0

                contorno = segmentation.mark_boundaries(image, copia,
                                                        color=(1, 0, 0))
                name = '../../data/output/rw_fuzzy_params/'+function+'/imagens/'+i+'-'+str(param)+'-contorno.jpg'
                scipy.misc.imsave(name, contorno)

                for x in range(0, image.shape[0]):
                    for y in range(0, image.shape[1]):
                        val1 = oct_cells[x, y]
                        val2 = copia[x, y]

                        if val1 == 1:
                            markers[x][y] = 1
                        if val2 == 0:
                            markers[x][y] = 2

                ouro = io.imread('../../seeds/imagens/'+i+'_bin.bmp')

                labels_rw = random_walker(image,
                                          markers,
                                          beta=10000, tol=0.02, mode='bf')
                contorno = \
                    segmentation.mark_boundaries(image, ouro,
                                                 color=(0, 0, 0))
                contorno = \
                    segmentation.mark_boundaries(contorno,
                                                 labels_rw,
                                                 color=(0, 1, 0))
                name = \
                    '../../data/output/rw_fuzzy_params/'+function+'/imagens/'+i+'-'+str(param)+'.jpg'
                scipy.misc.imsave(name, contorno)
                labels_rw = (2-labels_rw)
                TP = sum((labels_rw == 1) & (ouro == 255))
                somaTP = sum(TP)
                TN = sum((labels_rw == 0) & (ouro == 0))
                somaTN = sum(TN)
                FN = sum((labels_rw == 0) & (ouro == 255))

                FP = sum((labels_rw == 1) & (ouro == 0))
                somaFN = sum(FN)
                somaFP = sum(FP)

                linha.append(str(somaTP))
                soma[param][pos] += somaTP
                pos += 1
                linha.append(str(somaTN))
                soma[param][pos] += somaTN
                pos += 1
                linha.append(str(somaFP))
                soma[param][pos] += somaFP
                pos += 1
                linha.append(str(somaFN))
                soma[param][pos] += somaFN
                pos += 1

                XOR = (somaFP + somaFN)/(somaTP + somaFN)
                Precision = somaTP/float(somaTP + somaFP)
                Sensitivity = somaTP/float(somaTP + somaFN)
                Specificity = somaTN/float(somaFP + somaTN)
                Jaccard = somaTP/float(somaTP + somaFN + somaFP)
                Recall = somaTP/float(somaTP + somaFN)

                if (Precision != 0) and (Recall != 0):
                    Fmeasure = \
                        (Precision*Recall)/(Precision+Recall)
                else:
                    Fmeasure = 0

                linha.append(str(Sensitivity))
                soma[param][pos] += Sensitivity
                pos += 1
                linha.append(str(Specificity))
                soma[param][pos] += Specificity
                pos += 1
                linha.append(str(Jaccard))
                soma[param][pos] += Jaccard
                pos += 1
                linha.append(str(1-Specificity))
                soma[param][pos] += 1-Specificity
                pos += 1
            output.writerow(linha)

    for param in params['params']:
        for valor in soma[param]:
            media[param].append(str(valor/len(images)))
        media[param][0] = seeds
        medias[param].append(media)

    with open('../../data/output/rw_fuzzy_params/'+function+'/result_medias.csv',
              "w") as result_medias:
        output_2 = csv.writer(result_medias, quoting=csv.QUOTE_ALL)

        cabecalho = ['']
        cabecalho.append('TP')
        cabecalho.append('TN')
        cabecalho.append('FP')
        cabecalho.append('FN')
        cabecalho.append('Sensibility')
        cabecalho.append('Especificy')
        cabecalho.append('Jaccard')
        cabecalho.append('TFP')
        output_2.writerow(cabecalho)
        for a in params['params']:
            pos_2 = 1
            linha = []
            linha.append(params['name']+' ='+str(a))
            for t in range(pos_2, pos_2+8):
                linha.append(media[a][t])
            pos_2 += 8
            output_2.writerow(linha)
