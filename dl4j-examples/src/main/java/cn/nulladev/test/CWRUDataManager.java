package cn.nulladev.test;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

public class CWRUDataManager {

    public static List<CWRUData> dataList = new ArrayList();

    public static DataSet genAnnDataSet() {
        List<DataSet> sets = new ArrayList();
        for (var data: dataList) {
            if (data.pos != 0 && data.pos != 6) continue;
            if (data.depth == 28) continue;
            INDArray[] input = data.blocks.stream().map(d->Nd4j.create(d.DE, 1, 512)).toArray(INDArray[]::new);
            INDArray inputs = Nd4j.vstack(input);
            INDArray outputs = Nd4j.hstack(Nd4j.zeros(inputs.rows(), data.type()), Nd4j.ones(inputs.rows(), 1), Nd4j.zeros(inputs.rows(), 9-data.type()));
            DataSet set = new DataSet(inputs, outputs);
            sets.add(set);
        }
        DataSet dataSet = DataSet.merge(sets);
        dataSet.shuffle();
        return dataSet;
    }

    public static void AnnData() {
        int[] typeCount = new int[10];
        for (var data: dataList) {
            if (data.pos != 0 && data.pos != 6) continue;
            if (data.depth == 28) continue;
            typeCount[data.type()] += data.blocks.size();
            System.out.println(data.type() + ":" + data.name);
        }
        for (int i = 0; i<10; i++) {
            System.out.println(i + ":" + typeCount[i]);
        }
    }

    public static DataSet genAEDataSet() {
        List<DataSet> sets = new ArrayList();
        for (var data: dataList) {
            if (data.pos != 0 && data.pos != 3) continue;
            INDArray[] input = data.blocks.stream().map(d->Nd4j.create(d.DE, 1, 512)).toArray(INDArray[]::new);
            INDArray inputs = Nd4j.vstack(input);
            DataSet set = new DataSet(inputs, inputs);
            sets.add(set);
        }
        return DataSet.merge(sets);
    }

}
