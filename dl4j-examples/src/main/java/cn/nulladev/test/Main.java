package cn.nulladev.test;

import cn.nulladev.util.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

public class Main {

    public static final Random random = new Random();

    public static void main(String[] Args) {
        try {
            CWRUDataParser.parse();
            //CWRUDataManager.dataList.forEach(d->d.cut(512, 400));
            CWRUDataManager.dataList.forEach(d->d.cut(2048, 200));

            SplitTestAndTrain data = CWRUDataManager.genGenericDataSet1().splitTestAndTrain(0.9);
            //SplitTestAndTrain data = CWRUDataManager.genDCTDataSet().splitTestAndTrain(0.9);
            DataSet trainingData = data.getTrain();
            DataSet testData = data.getTest();

            /*
            int[] count = new int[10];
            for (int i = 0; i< trainingData.getLabels().rows(); i++) {
                for (int j = 0; j< 10; j++) {
                    if (trainingData.getLabels().getDouble(i ,j) == 1)
                        count[j] += 1;
                }
            }
            for (int j = 0; j< 10; j++) {
                System.out.println(count[j]);
            }
             */

/*
            INDArray feature = trainingData.getFeatures().getRow((long) (random.nextDouble()*trainingData.getFeatures().rows()));
            Visualizer.INDArray2IMG(feature, "E:/Temp/vis-"+0+".png");
            INDArray dct = Nd4j.create(DCT.dct(feature.toDoubleVector()), 1, 512);
            Visualizer.INDArray2IMG(dct, "E:/Temp/vis-dct-"+0+".png");

            INDArray idct = Nd4j.create(DCT.idct(dct.toDoubleVector()), 1, 512);
            Visualizer.INDArray2IMG(idct, "E:/Temp/vis-idct-"+0+".png");
*/

/*
            INDArray fft = Nd4j.create(Complex.absArray(FFT.fft(feature.toDoubleVector())), 1, 512);
            Visualizer.INDArray2IMG(fft, "E:/Temp/vis-fft-"+0+".png");

            INDArray ifft = Nd4j.create(Complex.absArray(FFT.ifft(fft.toDoubleVector())), 1, 512);
            Visualizer.INDArray2IMG(ifft, "E:/Temp/vis-ifft-"+0+".png");
*/

            Trainer.CNN(trainingData, testData);
            //Trainer.ANNResult(trainingData, testData);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
