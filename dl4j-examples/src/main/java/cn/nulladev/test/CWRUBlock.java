package cn.nulladev.test;

public class CWRUBlock {

    public final int size;
    public final CWRUData source;
    public double[] DE;
    public double[] FE;
    public double[] BA;

    public CWRUBlock(int size, CWRUData source) {
        this.size = size;
        this.source = source;
    }
}
