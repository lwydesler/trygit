import xml.etree.ElementTree as ET
import gdal
import torch
import numpy as np

xml = "Z:/ding/beijing/KX10_GIU_20220103_E116.94_N39.61_202200111720_L4A/KX10_GIU_20220103_E116.94_N39.61_202200111720_L4A.calib.xml"
with open(xml, "r", encoding="GB2312") as f:
    datasource = f.read()
    f.close()

tree = ET.fromstring(datasource)
print(tree)
print(tree.findall('RADIOMETRIC_CALIBRATION'))
x = tree.findall('RADIOMETRIC_CALIBRATION')
print(x[0][1][3])

RGB_GAIN = [x[0][1][3][0].text, x[0][1][3][2].text, x[0][1][3][4].text]
RGB_BIAS = [x[0][1][3][1].text, x[0][1][3][3].text, x[0][1][3][5].text]

LH_GAIN = [x[0][1][3][6].text, x[0][1][3][8].text]
LH_BIAS = [x[0][1][3][7].text, x[0][1][3][9].text]

"1-500"
TIS_GAIN1 = [x[0][2][3][0].text, x[0][2][3][3].text, x[0][2][3][6].text]
TIS_BIAS1 = [x[0][2][3][1].text, x[0][2][3][4].text, x[0][2][3][7].text]
TIS_BG1 = [x[0][2][3][2].text, x[0][2][3][5].text, x[0][2][3][8].text]

"501-975"
TIS_GAIN2 = [x[0][2][7][0].text, x[0][2][7][3].text, x[0][2][7][6].text]
TIS_BIAS2 = [x[0][2][7][1].text, x[0][2][7][4].text, x[0][2][7][7].text]
TIS_BG2 = [x[0][2][7][2].text, x[0][2][7][5].text, x[0][2][7][8].text]

"976-1475"
TIS_GAIN3 = [x[0][2][11][0].text, x[0][2][11][3].text, x[0][2][11][6].text]
TIS_BIAS3 = [x[0][2][11][1].text, x[0][2][11][4].text, x[0][2][11][7].text]
TIS_BG3 = [x[0][2][11][2].text, x[0][2][11][5].text, x[0][2][11][8].text]

"1476-1987"
TIS_GAIN4 = [x[0][2][15][0].text, x[0][2][15][3].text, x[0][2][15][6].text]
TIS_BIAS4 = [x[0][2][15][1].text, x[0][2][15][4].text, x[0][2][15][7].text]
TIS_BG4 = [x[0][2][15][2].text, x[0][2][15][5].text, x[0][2][15][8].text]

print(TIS_GAIN3[0])
print(int(TIS_GAIN3[0]))
print(TIS_BIAS3)

def get_calib_file(path, model):

    with open(path, "r", encoding="GB2312") as f:
        datasource = f.read()
        f.close()

    tree = ET.fromstring(datasource)
    print(tree)
    print(tree.findall('RADIOMETRIC_CALIBRATION'))
    x = tree.findall('RADIOMETRIC_CALIBRATION')
    print(x[0][1][3])

    if model == 'GIU':

        RGB_GAIN = [x[0][1][3][0].text, x[0][1][3][2].text, x[0][1][3][4].text]
        RGB_BIAS = [x[0][1][3][1].text, x[0][1][3][3].text, x[0][1][3][5].text]

        LH_GAIN = [x[0][1][3][6].text, x[0][1][3][8].text, (x[0][1][3][6].text+x[0][1][3][8].text)/2]
        LH_BIAS = [x[0][1][3][7].text, x[0][1][3][9].text, (x[0][1][3][7].text+x[0][1][3][9].text)/2]

        GAIN = [RGB_GAIN, LH_GAIN]
        BIAS = [RGB_BIAS, LH_BIAS]

        return GAIN, BIAS

    if model == 'TIS':

        "1-500"
        TIS_GAIN1 = [x[0][2][3][0].text, x[0][2][3][3].text, x[0][2][3][6].text]
        TIS_BIAS1 = [x[0][2][3][1].text, x[0][2][3][4].text, x[0][2][3][7].text]
        TIS_BG1 = [x[0][2][3][2].text, x[0][2][3][5].text, x[0][2][3][8].text]

        "501-975"
        TIS_GAIN2 = [x[0][2][7][0].text, x[0][2][7][3].text, x[0][2][7][6].text]
        TIS_BIAS2 = [x[0][2][7][1].text, x[0][2][7][4].text, x[0][2][7][7].text]
        TIS_BG2 = [x[0][2][7][2].text, x[0][2][7][5].text, x[0][2][7][8].text]

        "976-1475"
        TIS_GAIN3 = [x[0][2][11][0].text, x[0][2][11][3].text, x[0][2][11][6].text]
        TIS_BIAS3 = [x[0][2][11][1].text, x[0][2][11][4].text, x[0][2][11][7].text]
        TIS_BG3 = [x[0][2][11][2].text, x[0][2][11][5].text, x[0][2][11][8].text]

        "1476-1987"
        TIS_GAIN4 = [x[0][2][15][0].text, x[0][2][15][3].text, x[0][2][15][6].text]
        TIS_BIAS4 = [x[0][2][15][1].text, x[0][2][15][4].text, x[0][2][15][7].text]
        TIS_BG4 = [x[0][2][15][2].text, x[0][2][15][5].text, x[0][2][15][8].text]

        GAIN = [TIS_GAIN1, TIS_GAIN2, TIS_GAIN3, TIS_GAIN4]
        BIAS = [TIS_BIAS1, TIS_BIAS2, TIS_BIAS3, TIS_BIAS4]
        BG = [TIS_BG1, TIS_BG2, TIS_BG3, TIS_BG4]

        return GAIN, BIAS, BG

def getGeofiletoNumpy(file):

    alt = gdal.Open(file)
    cols = alt.RasterXSize
    rows = alt.RasterYSize
    dataarray = alt.ReadAsArray(0, 0, cols, rows).astype(np.float)

    return dataarray

def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()

    return uu

def Glimmer_Radiation_calibration(arrayRGB, arrayLH, GAIN, BIAS):
    DNRGB = GAIN[0]
    BIASRGB = BIAS[0]

    DNLH = GAIN[1]
    BIASLH = BIAS[1]

    for i in range(0, len(DNRGB)):
        arrayRGB[i]*DNRGB[i]+BIASRGB[i]

    for i in range(0, len(DNLH)):
        arrayLH[i]*DNLH[i]+BIASLH[i]

    return arrayRGB, arrayLH

def Thermal_Radiation_calibration(array, GAIN, BIAS, BG):
    DN1 = GAIN[0]
    BIAS1 = BIAS[0]
    Back_Ground1 = BG[0]
    DN2 = GAIN[1]
    BIAS2 = BIAS[1]
    Back_Ground2 = BG[1]
    DN3 = GAIN[2]
    BIAS3 = BIAS[2]
    Back_Ground3 = BG[2]
    DN4 = GAIN[3]
    BIAS4 = BIAS[3]
    Back_Ground4 = BG[3]

    for i in range(0, len(DN1)):

        array[i, 0:500, :] * DN1[i] + BIAS1[i]-Back_Ground1[i]
        array[i, 500:975, :] * DN2[i] + BIAS2[i] - Back_Ground2[i]
        array[i, 975:1475, :] * DN3[i] + BIAS3[i] - Back_Ground3[i]
        array[i, 1475:1987, :] * DN4[i] + BIAS4[i] - Back_Ground4[i]

    return array

#for GIU in tree.findall('GIU'):
#    print(GIU)

#GIU = tree.findall('GIU')
#print(GIU[0])
#RADIANCE_GAIN_BAND_1 = GIU.find('RADIANCE_GAIN_BAND_1').text





