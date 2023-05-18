import gdal
import numpy as np
import xml.etree.ElementTree as ET
import math
class calibration_SDG:

    def __init__(self, calibrate_file):
        """
        :param calibrate_file: path of  calibrate_file, xml
        :param model: optional paramter, "GIU", "TIS"
        """
        self.calibrate_file = calibrate_file

    def getGeofiletoNumpy(self, file):

        alt = gdal.Open(file)
        cols = alt.RasterXSize
        rows = alt.RasterYSize
        dataarray = alt.ReadAsArray(0, 0, cols, rows).astype(np.float)

        return dataarray

    def GetpredictImage(self, alt_adress, pre_data, pre_adress):
        alt = gdal.Open(alt_adress)
        cols = alt.RasterXSize
        rows = alt.RasterYSize
        im_width = cols  # 栅格矩阵的列数
        im_height = rows  # 栅格矩阵的行数

        outbandsize = alt.RasterCount

        a, a1, b, c, d, a2 = alt.GetGeoTransform()  # 仿射矩阵
        #print(a, a1, b, c, d, a2)
        im_geotrans = (a, a1, b, c, d, a2)  # 仿射矩阵

        im_proj = alt.GetProjection()  # 地图投影信息
        im_data = alt.ReadAsArray(0, 0, im_width, im_height)

        _, im_height, im_width = im_data.shape

        driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
        dataset1 = driver.Create(pre_adress, im_width, im_height, outbandsize, gdal.GDT_Float64)
        dataset1.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset1.SetProjection(im_proj)  # 写入投影

        for i in range(0, outbandsize):
            dataset1.GetRasterBand(i+1).WriteArray(pre_data[i])
        del dataset1

    def get_calib_file(self, path, model):

        with open(path, "r", encoding="GB2312") as f:
            datasource = f.read()
            f.close()

        tree = ET.fromstring(datasource)
        #print(tree)
        #print(tree.findall('RADIOMETRIC_CALIBRATION'))
        x = tree.findall('RADIOMETRIC_CALIBRATION')
        #print(x[0][1][3])

        if model == 'GIU':
            RGB_GAIN = [float(x[0][1][3][0].text), float(x[0][1][3][2].text), float(x[0][1][3][4].text)]
            RGB_BIAS = [float(x[0][1][3][1].text), float(x[0][1][3][3].text), float(x[0][1][3][5].text)]

            LH_GAIN = [float(x[0][1][3][6].text), float(x[0][1][3][8].text), (float(x[0][1][3][6].text) + float(x[0][1][3][8].text)) / 2]
            LH_BIAS = [float(x[0][1][3][7].text), float(x[0][1][3][9].text), (float(x[0][1][3][7].text) + float(x[0][1][3][9].text)) / 2]

            GAIN = [RGB_GAIN, LH_GAIN]
            BIAS = [RGB_BIAS, LH_BIAS]

            return GAIN, BIAS

        if model == 'TIS':
            "1-500"
            TIS_GAIN1 = [float(x[0][2][3][0].text), float(x[0][2][3][3].text), float(x[0][2][3][6].text)]
            TIS_BIAS1 = [float(x[0][2][3][1].text), float(x[0][2][3][4].text), float(x[0][2][3][7].text)]
            TIS_BG1 = [float(x[0][2][3][2].text), float(x[0][2][3][5].text), float(x[0][2][3][8].text)]

            "501-975"
            TIS_GAIN2 = [float(x[0][2][7][0].text), float(x[0][2][7][3].text), float(x[0][2][7][6].text)]
            TIS_BIAS2 = [float(x[0][2][7][1].text), float(x[0][2][7][4].text), float(x[0][2][7][7].text)]
            TIS_BG2 = [float(x[0][2][7][2].text), float(x[0][2][7][5].text), float(x[0][2][7][8].text)]

            "976-1475"
            TIS_GAIN3 = [float(x[0][2][11][0].text), float(x[0][2][11][3].text), float(x[0][2][11][6].text)]
            TIS_BIAS3 = [float(x[0][2][11][1].text), float(x[0][2][11][4].text), float(x[0][2][11][7].text)]
            TIS_BG3 = [float(x[0][2][11][2].text), float(x[0][2][11][5].text), float(x[0][2][11][8].text)]

            "1476-1987"
            TIS_GAIN4 = [float(x[0][2][15][0].text), float(x[0][2][15][3].text), float(x[0][2][15][6].text)]
            TIS_BIAS4 = [float(x[0][2][15][1].text), float(x[0][2][15][4].text), float(x[0][2][15][7].text)]
            TIS_BG4 = [float(x[0][2][15][2].text), float(x[0][2][15][5].text), float(x[0][2][15][8].text)]

            GAIN = [TIS_GAIN1, TIS_GAIN2, TIS_GAIN3, TIS_GAIN4]
            BIAS = [TIS_BIAS1, TIS_BIAS2, TIS_BIAS3, TIS_BIAS4]
            BG = [TIS_BG1, TIS_BG2, TIS_BG3, TIS_BG4]

            return GAIN, BIAS, BG

    def Glimmer_Radiation_calibration(self, arrayRGB, arrayLH, GAIN, BIAS):
        DNRGB = GAIN[0]
        BIASRGB = BIAS[0]

        DNLH = GAIN[1]
        BIASLH = BIAS[1]

        for i in range(0, len(DNRGB)):
            arrayRGB[i] * DNRGB[i] + BIASRGB[i]
            arrayRGB[i][arrayRGB[i] == BIASRGB[i]] = None

        for i in range(0, len(DNLH)):
            arrayLH[i] * DNLH[i] + BIASLH[i]
            arrayLH[i][arrayLH[i] == BIASLH[i]] = None

        return arrayRGB, arrayLH

    def Thermal_Radiation_calibration(self, array, GAIN, BIAS, BG):

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
            array[i, 0:500, :] * DN1[i] + BIAS1[i] - Back_Ground1[i]
            array[i, 0:500, :][array[i, 0:500, :] == BIAS1[i]- Back_Ground1[i]] = None
            array[i, 500:975, :] * DN2[i] + BIAS2[i] - Back_Ground2[i]
            array[i, 500:975, :][array[i, 500:975, :] == BIAS2[i] - Back_Ground2[i]] = None
            array[i, 975:1475, :] * DN3[i] + BIAS3[i] - Back_Ground3[i]
            array[i, 975:1475, :][array[i, 975:1475, :] == BIAS3[i] - Back_Ground3[i]] = None
            array[i, 1475:1987, :] * DN4[i] + BIAS4[i] - Back_Ground4[i]
            array[i, 1475:1987, :][array[i, 1475:1987, :] == BIAS4[i] - Back_Ground4[i]] = None

        return array

    def calibrate_GIU(self, RGB_file, LH_file):

        new_RGB_file = RGB_file[:-4] + '_cali1.tiff'
        new_LH_file = LH_file[:-4] + '_cali1.tiff'

        RGB_array = self.getGeofiletoNumpy(RGB_file)
        LH_array = self.getGeofiletoNumpy(LH_file)
        print('stage 1:load')
        GAIN, BIAS = self.get_calib_file(self.calibrate_file, model='GIU')
        print('stage 2:read')
        arrayRGB, arrayLH = self.Glimmer_Radiation_calibration(RGB_array, LH_array, GAIN, BIAS)
        print('stage 3:caculated')
        self.GetpredictImage(RGB_file, arrayRGB, new_RGB_file)
        self.GetpredictImage(LH_file, arrayLH, new_LH_file)
        print('stage 4:output')

    def calibrate_TIS(self, TIS_file):

        new_TIS_file = TIS_file[:-5] + '_cali.tiff'

        TIS_array = self.getGeofiletoNumpy(TIS_file)
        print('stage 1:load')
        GAIN, BIAS, BG = self.get_calib_file(self.calibrate_file, model='TIS')
        print('stage 2:read')
        arrayTIS = self.Thermal_Radiation_calibration(TIS_array, GAIN, BIAS, BG)
        print('stage 3:caculated')
        self.GetpredictImage(TIS_file, arrayTIS, new_TIS_file)
        print('stage 4:output')

    def transform_to_temperature(self, array):

        B = [9.35, 10.73, 11.72]
        for i in range(0, len(B)):
            array[i] = (1E6*6.626E-34/((1.3806E-23)*B[i]))/(math.log(2E24*6.626E-34*2.9979E8**2/(array[i]*B[i]**5+1), math.e))

        return array

if __name__=="__main__":

    cali_file1 = 'Z:/ding/beijing/KX10_GIU_20220103_E116.94_N39.61_202200111720_L4A/KX10_GIU_20220103_E116.94_N39.61_202200111720_L4A.calib.xml'
    rgbpath1 = 'Z:/ding/beijing/KX10_GIU_20220103_E116.94_N39.61_202200111720_L4A/KX10_GIU_20220103_E116.94_N39.61_202200111720_L4A_A_RGB.tif'
    rgbpath2 = 'Z:/ding/beijing/KX10_GIU_20220103_E116.94_N39.61_202200111720_L4A/KX10_GIU_20220103_E116.94_N39.61_202200111720_L4A_B_RGB.tif'

    lhpath1 = 'Z:/ding/beijing/KX10_GIU_20220103_E116.94_N39.61_202200111720_L4A/KX10_GIU_20220103_E116.94_N39.61_202200111720_L4A_A_LH.tif'
    lhpath2 = 'Z:/ding/beijing/KX10_GIU_20220103_E116.94_N39.61_202200111720_L4A/KX10_GIU_20220103_E116.94_N39.61_202200111720_L4A_B_LH.tif'
    process = calibration_SDG(cali_file1)
    process.calibrate_GIU(rgbpath1, lhpath1)
    process.calibrate_GIU(rgbpath2, lhpath2)

    cali_file2 = 'Z:/ding/beijing/KX10_TIS_20220103_E116.60_N40.66_202200112575_L4A/KX10_TIS_20220103_E116.60_N40.66_202200112575_L4A.calib.xml'
    tis_file = 'Z:/ding/beijing/KX10_TIS_20220103_E116.60_N40.66_202200112575_L4A/KX10_TIS_20220103_E116.60_N40.66_202200112575_L4A.tiff'
    #process = calibration_SDG(cali_file2)
    #process.calibrate_TIS(tis_file)