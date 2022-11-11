package cn.nulladev.test;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;

import java.util.Random;

public class Main {

    public static final Random random = new Random();

    public static void main(String[] Args) {
        try {
            CWRUDataParser.parse();
            CWRUDataManager.dataList.forEach(d->d.cut(512, 400));
            //CWRUDataManager.dataList.forEach(d->d.cut(256, 1000));

            //SplitTestAndTrain data = CWRUDataManager.genGenericDataSet1().splitTestAndTrain(0.9);
            SplitTestAndTrain data = CWRUDataManager.genAEDataSet().splitTestAndTrain(0.9);
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

            INDArray fft = Nd4j.create(Complex.absArray(FFT.fft(feature.toDoubleVector())), 1, 512);
            Visualizer.INDArray2IMG(fft, "E:/Temp/vis-fft-"+0+".png");

            INDArray ifft = Nd4j.create(Complex.absArray(FFT.ifft(fft.toDoubleVector())), 1, 512);
            Visualizer.INDArray2IMG(ifft, "E:/Temp/vis-ifft-"+0+".png");
             */


            Trainer.AE(trainingData, testData);
            //Trainer.ANNResult(trainingData, testData);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
