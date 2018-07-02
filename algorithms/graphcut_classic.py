from constants import images
from skimage import io
from skimage import segmentation
import cv2
import csv
import numpy as np
import scipy.io
import scipy.io as sio
import scipy.misc
from utils.graphcut import build_bayes_graph


def run():
    seeds = 6
    medias = []
    with open('../data/output/classico/grab_rect/resultado.csv', "w") as resultado:
        output = csv.writer(resultado, quoting=csv.QUOTE_ALL)

        cabecalho_1 = []
        cabecalho_1.append("")
        titulo = str('')
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

        soma = [0] * cont
        for i in images:
            image = cv2.imread('../seeds/imagens/'+i+'.bmp')
            markers = np.zeros((image.shape[0], image.shape[1]))
            mat_contents = sio.loadmat('../seeds/labels_roi/'+i+'.mat')
            oct_cells = mat_contents['labels']
            val = oct_cells[0, 1]
            markers = np.zeros((image.shape[0], image.shape[1]), np.uint8)

            for x in range(0, image.shape[0]):
                for y in range(0, image.shape[1]):
                    val = oct_cells[x, y]

                    if val == 1:
                        markers[x][y] = 0
                    if val == -1:
                        markers[x][y] = 2

            ouro = io.imread('../seeds/imagens/'+i+'_bin.bmp')

            linha = []
            linha.append(i)

            pos = 0
            labels_rw = build_bayes_graph(
                im=image, labels=markers)

            contorno = \
                segmentation.mark_boundaries(image, ouro,
                                             color=(0, 0, 0))
            contorno = \
                segmentation.mark_boundaries(contorno,
                                             labels_rw[0],
                                             color=(0, 1, 0))
            name = \
                '../data/output/classico/grab_rect/images/'+i+'.jpg'
            scipy.misc.imsave(name, contorno)

            labels_rw = (2-labels_rw[0])

            TP = sum((labels_rw == 1) & (ouro == 255))
            somaTP = sum(TP)
            TN = sum((labels_rw == 0) & (ouro == 0))
            somaTN = sum(TN)
            FN = sum((labels_rw == 0) & (ouro == 255))

            FP = sum((labels_rw == 1) & (ouro == 0))
            somaFN = sum(FN)
            somaFP = sum(FP)

            linha.append(str(somaTP))
            soma[pos] += somaTP
            pos += 1
            linha.append(str(somaTN))
            soma[pos] += somaTN
            pos += 1
            linha.append(str(somaFP))
            soma[pos] += somaFP
            pos += 1
            linha.append(str(somaFN))
            soma[pos] += somaFN
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
            soma[pos] += Sensitivity
            pos += 1
            linha.append(str(Specificity))
            soma[pos] += Specificity
            pos += 1
            linha.append(str(Jaccard))
            soma[pos] += Jaccard
            pos += 1
            linha.append(str(1-Specificity))
            soma[pos] += 1-Specificity
            pos += 1
            output.writerow(linha)
            media = ['media']
            for valor in soma:
                media.append(str(valor/len(images)))
            media[0] = seeds
            medias.append(media)

            with open('../data/output/classico/grab_rect/result_medias.csv',
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
                pos_2 = 1
                linha = []
                linha.append('')
                for t in range(pos_2, pos_2+8):
                    linha.append(media[t])
                pos_2 += 8
                output_2.writerow(linha)


run()
