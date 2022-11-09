package cn.nulladev.test;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;

public class Main {

    public static void main(String[] Args) {
        try {
            CWRUDataParser.parse();
            CWRUDataManager.dataList.forEach(d->d.cut(512, 400));

            //CWRUDataManager.AnnData();

            SplitTestAndTrain data = CWRUDataManager.genAnnDataSet().splitTestAndTrain(0.9);
            //SplitTestAndTrain data = CWRUDataManager.genAEDataSet().splitTestAndTrain(0.9);
            DataSet trainingData = data.getTrain();
            DataSet testData = data.getTest();

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

            //Visualizer.INDArray2IMG(trainingData.getFeatures(), "E:/Temp/vis-"+0+".png");

            //Trainer.ANN(trainingData, testData);
            //Trainer.ANNResult(trainingData, testData);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
