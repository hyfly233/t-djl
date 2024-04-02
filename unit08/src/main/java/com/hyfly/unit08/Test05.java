package com.hyfly.unit08;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.util.DownloadUtils;
import ai.djl.util.Pair;
import com.hyfly.utils.timemachine.Vocab;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.HistogramTrace;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.stream.IntStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

public class Test05 {

    public static void main(String[] args) throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            String rawText = readDataNMT();
            System.out.println(rawText.substring(0, 75));

            String text = preprocessNMT(rawText);
            System.out.println(text.substring(0, 80));

            Pair<ArrayList<String[]>, ArrayList<String[]>> pair = tokenizeNMT(text.toString(), null);
            ArrayList<String[]> source = pair.getKey();
            ArrayList<String[]> target = pair.getValue();
            for (String[] subArr : source.subList(0, 6)) {
                System.out.println(Arrays.toString(subArr));
            }

            for (String[] subArr : target.subList(0, 6)) {
                System.out.println(Arrays.toString(subArr));
            }

            //
            double[] y1 = new double[source.size()];
            for (int i = 0; i < source.size(); i++) y1[i] = source.get(i).length;
            double[] y2 = new double[target.size()];
            for (int i = 0; i < target.size(); i++) y2[i] = target.get(i).length;

            HistogramTrace trace1 =
                    HistogramTrace.builder(y1).opacity(.75).name("source").nBinsX(20).build();
            HistogramTrace trace2 =
                    HistogramTrace.builder(y2).opacity(.75).name("target").nBinsX(20).build();

            Layout layout = Layout.builder().barMode(Layout.BarMode.GROUP).build();
            new Figure(layout, trace1, trace2);

            //
            Vocab srcVocab = new Vocab(
                    source.stream().toArray(String[][]::new),
                    2,
                    new String[]{"<pad>", "<bos>", "<eos>"});
            System.out.println(srcVocab.length());

            //
            int[] result = truncatePad(srcVocab.getIdxs(source.get(0)), 10, srcVocab.getIdx("<pad>"));
            System.out.println(Arrays.toString(result));

            //
            Pair<ArrayDataset, Pair<Vocab, Vocab>> output = loadDataNMT(2, 8, 600);
            ArrayDataset dataset = output.getKey();
            srcVocab = output.getValue().getKey();
            Vocab tgtVocab = output.getValue().getValue();

            Batch batch = dataset.getData(manager).iterator().next();
            NDArray X = batch.getData().get(0);
            NDArray xValidLen = batch.getData().get(1);
            NDArray Y = batch.getLabels().get(0);
            NDArray yValidLen = batch.getLabels().get(1);
            System.out.println(X);
            System.out.println(xValidLen);
            System.out.println(Y);
            System.out.println(yValidLen);
        }
    }

    public static String readDataNMT() throws IOException {
        DownloadUtils.download(
                "http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip", "fra-eng.zip");
        ZipFile zipFile = new ZipFile(new File("fra-eng.zip"));
        Enumeration<? extends ZipEntry> entries = zipFile.entries();
        while (entries.hasMoreElements()) {
            ZipEntry entry = entries.nextElement();
            if (entry.getName().contains("fra.txt")) {
                InputStream stream = zipFile.getInputStream(entry);
                return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
            }
        }
        return null;
    }

    public static String preprocessNMT(String text) {
        // 使用空格替换不间断空格
        // 使用小写字母替换大写字母
        text = text.replace('\u202f', ' ').replaceAll("\\xa0", " ").toLowerCase();

        // 在单词和标点符号之间插入空格
        StringBuilder out = new StringBuilder();
        Character currChar;
        for (int i = 0; i < text.length(); i++) {
            currChar = text.charAt(i);
            if (i > 0 && noSpace(currChar, text.charAt(i - 1))) {
                out.append(' ');
            }
            out.append(currChar);
        }
        return out.toString();
    }

    public static boolean noSpace(Character currChar, Character prevChar) {
        /* Preprocess the English-French dataset. */
        return new HashSet<>(Arrays.asList(',', '.', '!', '?')).contains(currChar)
                && prevChar != ' ';
    }

    public static Pair<ArrayList<String[]>, ArrayList<String[]>> tokenizeNMT(
            String text, Integer numExamples) {
        ArrayList<String[]> source = new ArrayList<>();
        ArrayList<String[]> target = new ArrayList<>();

        int i = 0;
        for (String line : text.split("\n")) {
            if (numExamples != null && i > numExamples) {
                break;
            }
            String[] parts = line.split("\t");
            if (parts.length == 2) {
                source.add(parts[0].split(" "));
                target.add(parts[1].split(" "));
            }
            i += 1;
        }
        return new Pair<>(source, target);
    }

    public static int[] truncatePad(Integer[] integerLine, int numSteps, int paddingToken) {
        // 截断或填充文本序列
        int[] line = Arrays.stream(integerLine).mapToInt(i -> i).toArray();
        if (line.length > numSteps) {
            return Arrays.copyOfRange(line, 0, numSteps);
        }
        int[] paddingTokenArr = new int[numSteps - line.length]; // Pad
        Arrays.fill(paddingTokenArr, paddingToken);

        return IntStream.concat(Arrays.stream(line), Arrays.stream(paddingTokenArr)).toArray();
    }

    public static Pair<NDArray, NDArray> buildArrayNMT(
            List<String[]> lines, Vocab vocab, int numSteps) {
        // 将机器翻译的文本序列转换成小批量
        List<Integer[]> linesIntArr = new ArrayList<>();
        for (String[] strings : lines) {
            linesIntArr.add(vocab.getIdxs(strings));
        }
        for (int i = 0; i < linesIntArr.size(); i++) {
            List<Integer> temp = new ArrayList<>(Arrays.asList(linesIntArr.get(i)));
            temp.add(vocab.getIdx("<eos>"));
            linesIntArr.set(i, temp.toArray(new Integer[0]));
        }

        NDManager manager = NDManager.newBaseManager();

        NDArray arr = manager.create(new Shape(linesIntArr.size(), numSteps), DataType.INT32);
        int row = 0;
        for (Integer[] line : linesIntArr) {
            NDArray rowArr = manager.create(truncatePad(line, numSteps, vocab.getIdx("<pad>")));
            arr.set(new NDIndex("{}:", row), rowArr);
            row += 1;
        }
        NDArray validLen = arr.neq(vocab.getIdx("<pad>")).sum(new int[]{1});
        return new Pair<>(arr, validLen);
    }

    public static Pair<ArrayDataset, Pair<Vocab, Vocab>> loadDataNMT(
            int batchSize, int numSteps, int numExamples) throws IOException {
        // 返回翻译数据集的迭代器和词表
        String text = preprocessNMT(readDataNMT());
        Pair<ArrayList<String[]>, ArrayList<String[]>> pair = tokenizeNMT(text, numExamples);
        ArrayList<String[]> source = pair.getKey();
        ArrayList<String[]> target = pair.getValue();
        Vocab srcVocab =
                new Vocab(
                        source.toArray(String[][]::new),
                        2,
                        new String[]{"<pad>", "<bos>", "<eos>"});
        Vocab tgtVocab =
                new Vocab(
                        target.toArray(String[][]::new),
                        2,
                        new String[]{"<pad>", "<bos>", "<eos>"});

        Pair<NDArray, NDArray> pairArr = buildArrayNMT(source, srcVocab, numSteps);
        NDArray srcArr = pairArr.getKey();
        NDArray srcValidLen = pairArr.getValue();

        pairArr = buildArrayNMT(target, tgtVocab, numSteps);
        NDArray tgtArr = pairArr.getKey();
        NDArray tgtValidLen = pairArr.getValue();

        ArrayDataset dataset =
                new ArrayDataset.Builder()
                        .setData(srcArr, srcValidLen)
                        .optLabels(tgtArr, tgtValidLen)
                        .setSampling(batchSize, true)
                        .build();

        return new Pair<>(dataset, new Pair<>(srcVocab, tgtVocab));
    }
}
