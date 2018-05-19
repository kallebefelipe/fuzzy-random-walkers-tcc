"""
Fuzzy witch normalizaton
"""

import numpy as np
from growcut import growcut_python
from skimage import io
import scipy.misc
from skimage import segmentation
import scipy.io as sio
import scipy.io
import csv
from datetime import datetime
from numpy import array

constante = [5]


medias = []
numero_marcacao = []
with open('output/resultado.csv', "w") as \
        resultado:
    output = csv.writer(resultado, quoting=csv.QUOTE_ALL)

    cabecalho_1 = []
    cabecalho_1.append("")

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

    images = ['mdb001', 'mdb002',  'mdb005', 'mdb010', 'mdb012',
              'mdb013', 'mdb015', 'mdb017', 'mdb019', 'mdb021',
              'mdb023', 'mdb025', 'mdb028', 'mdb030', 'mdb032',
              'mdb058', 'mdb063', 'mdb069', 'mdb080', 'mdb091',
              'mdb132_1', 'mdb132_2', 'mdb134', 'mdb141', 'mdb142',
              'mdb144_1', 'mdb144_2', 'mdb145', 'mdb148', 'mdb175',
              'mdb178', 'mdb179', 'mdb181', 'mdb184', 'mdb186',
              'mdb188', 'mdb190', 'mdb191', 'mdb193', 'mdb195',
              'mdb198', 'mdb199', 'mdb202', 'mdb204', 'mdb206',
              'mdb207', 'mdb244', 'mdb264', 'mdb265', 'mdb267',
              'mdb270', 'mdb271', 'mdb274', 'mdb290', 'mdb312',
              'mdb314', 'mdb315']
    images = ['mdb312']
    soma = [0] * cont
    for count, i in enumerate(images):
        image = io.imread('data/imagens/'+i+'.bmp')
        count = 0
        mat_contents = sio.loadmat('data/labels_roi/'+i+'.mat')
        oct_cells = mat_contents['labels']
        val = oct_cells[0, 1]
        markers = np.zeros((image.shape[0], image.shape[1], 2))

        for x in range(0, image.shape[0]):
            for y in range(0, image.shape[1]):
                val = oct_cells[x, y]

                if val == 1:
                    markers[x][y][0] = 1
                    markers[x][y][1] = 1
                if val == -1:
                    markers[x][y][0] = -1
                    markers[x][y][1] = -1

        ouro = io.imread('data/imagens/'+i+'_bin.bmp')

        linha = []
        linha.append(i)

        pos = 0
        inicio = datetime.now()
        labels_rw = growcut_python(image,
                                   markers)
        fim = datetime.now()
        print(str(count)+' ' + str(fim - inicio))
        new_labels = []
        for label_i in range(0, len(labels_rw)):
            new_col = []
            for label_j in range(0, len(labels_rw[label_i])):
                new_col.append(int(labels_rw[label_i][label_j]))
            new_labels.append(new_col)
        import ipdb; ipdb.set_trace()
        contorno = \
            segmentation.mark_boundaries(image, ouro,
                                         color=(0, 0, 0))
        contorno = \
            segmentation.mark_boundaries(contorno,
                                         array(new_labels),
                                         color=(0, 1, 0))
        name = 'output/imagens/'+i+'.jpg'
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
    medias.append(media)
    output.writerow(media)

    with open('output/result_medias.csv',
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

with open('output/resultados.csv', "w") as output:
    saida = csv.writer(output, quoting=csv.QUOTE_ALL)
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

    saida.writerow(cabecalho_1)

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
    saida.writerow(cabecalho_2)

    for each in medias:
        saida.writerow(each)
