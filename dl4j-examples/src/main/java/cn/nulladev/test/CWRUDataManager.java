package cn.nulladev.test;

import cn.nulladev.util.Complex;
import cn.nulladev.util.DCT;
import cn.nulladev.util.FFT;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class CWRUDataManager {

    public static List<CWRUData> dataList = new ArrayList();

    public static DataSet genGenericDataSet() {
        List<DataSet> sets = new ArrayList();
        for (var data: dataList) {
            if (data.pos != 0 && data.pos != 6) continue;
            if (data.depth == 28) continue;
            INDArray[] input = data.blocks.stream().map(d->Nd4j.create(d.DE, 1, d.DE.length)).toArray(INDArray[]::new);
            INDArray inputs = Nd4j.vstack(input);
            INDArray outputs = Nd4j.hstack(Nd4j.zeros(inputs.rows(), data.type()), Nd4j.ones(inputs.rows(), 1), Nd4j.zeros(inputs.rows(), 9-data.type()));
            DataSet set = new DataSet(inputs, outputs);
            sets.add(set);
        }
        DataSet dataSet = DataSet.merge(sets);
        dataSet.shuffle();
        return dataSet;
    }

    public static DataSet genGenericDataSet1() {
        List<CWRUBlock> blocks = new ArrayList();
        for (var data: dataList) {
            if (data.pos != 0 && data.pos != 6) continue;
            if (data.depth == 28) continue;
            data.blocks.forEach(blocks::add);
        }
        Collections.shuffle(blocks);
        INDArray[] input = blocks.stream().map(b->Nd4j.create(b.DE, 1, b.DE.length)).toArray(INDArray[]::new);
        INDArray inputs = Nd4j.vstack(input);
        INDArray[] output = blocks.stream().map(b->genOutputFromType(b.source.type())).toArray(INDArray[]::new);
        INDArray outputs = Nd4j.vstack(output);
        DataSet dataSet = new DataSet(inputs, outputs);
        return dataSet;
    }

    public static DataSet genFourierDataSet() {
        List<CWRUBlock> blocks = new ArrayList();
        for (var data: dataList) {
            if (data.pos != 0 && data.pos != 6) continue;
            if (data.depth == 28) continue;
            data.blocks.forEach(blocks::add);
        }
        Collections.shuffle(blocks);
        INDArray[] input = blocks.stream().map(b->Nd4j.create(Complex.absArray(FFT.fft(b.DE)), 1, b.DE.length)).toArray(INDArray[]::new);
        INDArray inputs = Nd4j.vstack(input);
        INDArray[] output = blocks.stream().map(b->genOutputFromType(b.source.type())).toArray(INDArray[]::new);
        INDArray outputs = Nd4j.vstack(output);
        DataSet dataSet = new DataSet(inputs, outputs);
        return dataSet;
    }

    public static DataSet genDCTDataSet() {
        List<CWRUBlock> blocks = new ArrayList();
        for (var data: dataList) {
            if (data.pos != 0 && data.pos != 6) continue;
            if (data.depth == 28) continue;
            data.blocks.forEach(blocks::add);
        }
        Collections.shuffle(blocks);
        INDArray[] input = blocks.stream().map(b->Nd4j.create(DCT.dct(b.DE), 1, b.DE.length)).toArray(INDArray[]::new);
        INDArray inputs = Nd4j.vstack(input);
        INDArray[] output = blocks.stream().map(b->genOutputFromType(b.source.type())).toArray(INDArray[]::new);
        INDArray outputs = Nd4j.vstack(output);
        DataSet dataSet = new DataSet(inputs, outputs);
        return dataSet;
    }

    public static INDArray genOutputFromType(int type) {
        INDArray output = Nd4j.zeros(1, 10);
        output.putScalar(0, type, 1);
        return output;
    }

    public static void printSetInfo() {
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
        List<CWRUBlock> blocks = new ArrayList();
        for (var data: dataList) {
            if (data.pos != 0 && data.pos != 6) continue;
            if (data.depth == 28) continue;
            data.blocks.forEach(blocks::add);
        }
        Collections.shuffle(blocks);
        INDArray[] input = blocks.stream().map(b->Nd4j.create(b.DE, 1, b.DE.length)).toArray(INDArray[]::new);
        INDArray inputs = Nd4j.vstack(input);
        DataSet dataSet = new DataSet(inputs, inputs);
        return dataSet;
    }

}
