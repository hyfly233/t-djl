package com.hyfly.unit13;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import ai.djl.util.ZipUtils;
import com.hyfly.unit13.entity.RandomGenerator;
import com.hyfly.utils.timemachine.Vocab;
import lombok.extern.slf4j.Slf4j;
import tech.tablesaw.plotly.components.Axis;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.HistogramTrace;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@Slf4j
public class Test1403 {

    public static void main(String[] args) throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            String[][] sentences = readPTB();
            System.out.println("# sentences: " + sentences.length);

            Vocab vocab = new Vocab(sentences, 10, new String[]{});
            System.out.println(vocab.length());

            String[][] subsampled = subSampling(sentences, vocab);

            //
            double[] y1 = new double[sentences.length];
            for (int i = 0; i < sentences.length; i++) y1[i] = sentences[i].length;
            double[] y2 = new double[subsampled.length];
            for (int i = 0; i < subsampled.length; i++) y2[i] = subsampled[i].length;

            HistogramTrace trace1 =
                    HistogramTrace.builder(y1).opacity(.75).name("origin").nBinsX(20).build();
            HistogramTrace trace2 =
                    HistogramTrace.builder(y2).opacity(.75).name("subsampled").nBinsX(20).build();

            Layout layout =
                    Layout.builder()
                            .barMode(Layout.BarMode.GROUP)
                            .showLegend(true)
                            .xAxis(Axis.builder().title("# tokens per sentence").build())
                            .yAxis(Axis.builder().title("count").build())
                            .build();
            new Figure(layout, trace1, trace2);

            System.out.println(compareCounts("the", sentences, subsampled));
            System.out.println(compareCounts("join", sentences, subsampled));

            Integer[][] corpus = new Integer[subsampled.length][];
            for (int i = 0; i < subsampled.length; i++) {
                corpus[i] = vocab.getIdxs(subsampled[i]);
            }
            for (int i = 0; i < 3; i++) {
                System.out.println(Arrays.toString(corpus[i]));
            }

            //
            Integer[][] tinyDataset =
                    new Integer[][]{
                            IntStream.range(0, 7)
                                    .boxed()
                                    .collect(Collectors.toList())
                                    .toArray(new Integer[]{}),
                            IntStream.range(7, 10)
                                    .boxed()
                                    .collect(Collectors.toList())
                                    .toArray(new Integer[]{})
                    };

            System.out.println("dataset " + Arrays.deepToString(tinyDataset));
            Pair<ArrayList<Integer>, ArrayList<ArrayList<Integer>>> centerContextPair =
                    getCentersAndContext(tinyDataset, 2);
            for (int i = 0; i < centerContextPair.getValue().size(); i++) {
                System.out.println(
                        "Center "
                                + centerContextPair.getKey().get(i)
                                + " has contexts"
                                + centerContextPair.getValue().get(i));
            }

            //
            centerContextPair = getCentersAndContext(corpus, 5);
            ArrayList<Integer> allCenters = centerContextPair.getKey();
            ArrayList<ArrayList<Integer>> allContexts = centerContextPair.getValue();
            System.out.println("中心词-上下文词对”的数量:" + allCenters.size());

            //
            RandomGenerator generator =
                    new RandomGenerator(Arrays.asList(new Double[]{2.0, 3.0, 4.0}));
            Integer[] generatorOutput = new Integer[10];
            for (int i = 0; i < 10; i++) {
                generatorOutput[i] = generator.draw();
            }
            System.out.println(Arrays.toString(generatorOutput));


            //
            ArrayList<ArrayList<Integer>> allNegatives = getNegatives(allContexts, corpus, 5);

            //
            NDList x1 =
                    new NDList(
                            manager.create(new int[]{1}),
                            manager.create(new int[]{2, 2}),
                            manager.create(new int[]{3, 3, 3, 3}));
            NDList x2 =
                    new NDList(
                            manager.create(new int[]{1}),
                            manager.create(new int[]{2, 2, 2}),
                            manager.create(new int[]{3, 3}));

            NDList batchedData = batchifyData(manager, new NDList[]{x1, x2});
            String[] names = new String[]{"centers", "contexts_negatives", "masks", "labels"};
            for (int i = 0; i < batchedData.size(); i++) {
                System.out.println(names[i] + " shape: " + batchedData.get(i));
            }

            //
            Pair<ArrayDataset, Vocab> datasetVocab = loadDataPTB(512, 5, 5, manager);
            ArrayDataset dataset = datasetVocab.getKey();
            vocab = datasetVocab.getValue();

            Batch batch = dataset.getData(manager).iterator().next();
            for (int i = 0; i < batch.getData().size(); i++) {
                System.out.println(names[i] + " shape: " + batch.getData().get(i).getShape());
            }
        }
    }

    public static String[][] readPTB() throws IOException {
        String ptbURL = "http://d2l-data.s3-accelerate.amazonaws.com/ptb.zip";
        InputStream input = new URL(ptbURL).openStream();
        ZipUtils.unzip(input, Paths.get("./"));

        ArrayList<String> lines = new ArrayList<>();
        File file = new File("./ptb/ptb.train.txt");
        Scanner myReader = new Scanner(file);
        while (myReader.hasNextLine()) {
            lines.add(myReader.nextLine());
        }
        String[][] tokens = new String[lines.size()][];
        for (int i = 0; i < lines.size(); i++) {
            tokens[i] = lines.get(i).trim().split(" ");
        }
        return tokens;
    }

    public static boolean keep(String token, LinkedHashMap<?, Integer> counter, int numTokens) {
        // Return True if to keep this token during subsampling
        return new Random().nextFloat() < Math.sqrt(1e-4 / counter.get(token) * numTokens);
    }

    public static String[][] subSampling(String[][] sentences, Vocab vocab) {
        for (int i = 0; i < sentences.length; i++) {
            for (int j = 0; j < sentences[i].length; j++) {
                sentences[i][j] = vocab.idxToToken.get(vocab.getIdx(sentences[i][j]));
            }
        }
        // Count the frequency for each word
        LinkedHashMap<?, Integer> counter = vocab.countCorpus2D(sentences);
        int numTokens = 0;
        for (Integer value : counter.values()) {
            numTokens += value;
        }

        // Now do the subsampling
        String[][] output = new String[sentences.length][];
        for (int i = 0; i < sentences.length; i++) {
            ArrayList<String> tks = new ArrayList<>();
            for (int j = 0; j < sentences[i].length; j++) {
                String tk = sentences[i][j];
                if (keep(sentences[i][j], counter, numTokens)) {
                    tks.add(tk);
                }
            }
            output[i] = tks.toArray(new String[tks.size()]);
        }

        return output;
    }

    public static String compareCounts(String token, String[][] sentences, String[][] subsampled) {
        int beforeCount = 0;
        for (int i = 0; i < sentences.length; i++) {
            for (int j = 0; j < sentences[i].length; j++) {
                if (sentences[i][j].equals(token)) beforeCount += 1;
            }
        }

        int afterCount = 0;
        for (int i = 0; i < subsampled.length; i++) {
            for (int j = 0; j < subsampled[i].length; j++) {
                if (subsampled[i][j].equals(token)) afterCount += 1;
            }
        }

        return "# of \"the\": before=" + beforeCount + ", after=" + afterCount;
    }

    public static Pair<ArrayList<Integer>, ArrayList<ArrayList<Integer>>> getCentersAndContext(
            Integer[][] corpus, int maxWindowSize) {
        ArrayList<Integer> centers = new ArrayList<>();
        ArrayList<ArrayList<Integer>> contexts = new ArrayList<>();

        for (Integer[] line : corpus) {
            // Each sentence needs at least 2 words to form a "central target word
            // - context word" pair
            if (line.length < 2) {
                continue;
            }
            centers.addAll(Arrays.asList(line));
            for (int i = 0; i < line.length; i++) { // Context window centered at i
                int windowSize = new Random().nextInt(maxWindowSize - 1) + 1;
                List<Integer> indices =
                        IntStream.range(
                                        Math.max(0, i - windowSize),
                                        Math.min(line.length, i + 1 + windowSize))
                                .boxed()
                                .collect(Collectors.toList());
                // Exclude the central target word from the context words
                indices.remove(indices.indexOf(i));
                ArrayList<Integer> context = new ArrayList<>();
                for (Integer idx : indices) {
                    context.add(line[idx]);
                }
                contexts.add(context);
            }
        }
        return new Pair<>(centers, contexts);
    }

    public static ArrayList<ArrayList<Integer>> getNegatives(
            ArrayList<ArrayList<Integer>> allContexts, Integer[][] corpus, int K) {
        LinkedHashMap<?, Integer> counter = Vocab.countCorpus2D(corpus);
        ArrayList<Double> samplingWeights = new ArrayList<>();
        for (Map.Entry<?, Integer> entry : counter.entrySet()) {
            samplingWeights.add(Math.pow(entry.getValue(), .75));
        }
        ArrayList<ArrayList<Integer>> allNegatives = new ArrayList<>();
        RandomGenerator generator = new RandomGenerator(samplingWeights);
        for (ArrayList<Integer> contexts : allContexts) {
            ArrayList<Integer> negatives = new ArrayList<>();
            while (negatives.size() < contexts.size() * K) {
                Integer neg = generator.draw();
                // Noise words cannot be context words
                if (!contexts.contains(neg)) {
                    negatives.add(neg);
                }
            }
            allNegatives.add(negatives);
        }
        return allNegatives;
    }

    public static NDList batchifyData(NDManager manager, NDList[] data) {
        NDList centers = new NDList();
        NDList contextsNegatives = new NDList();
        NDList masks = new NDList();
        NDList labels = new NDList();

        long maxLen = 0;
        for (NDList ndList : data) { // center, context, negative = ndList
            maxLen =
                    Math.max(
                            maxLen,
                            ndList.get(1).countNonzero().getLong()
                                    + ndList.get(2).countNonzero().getLong());
        }
        for (NDList ndList : data) { // center, context, negative = ndList
            NDArray center = ndList.get(0);
            NDArray context = ndList.get(1);
            NDArray negative = ndList.get(2);

            int count = 0;
            for (int i = 0; i < context.size(); i++) {
                // If a 0 is found, we want to stop adding these
                // values to NDArray
                if (context.get(i).getInt() == 0) {
                    break;
                }
                contextsNegatives.add(context.get(i).reshape(1));
                masks.add(manager.create(1).reshape(1));
                labels.add(manager.create(1).reshape(1));
                count += 1;
            }
            for (int i = 0; i < negative.size(); i++) {
                // If a 0 is found, we want to stop adding these
                // values to NDArray
                if (negative.get(i).getInt() == 0) {
                    break;
                }
                contextsNegatives.add(negative.get(i).reshape(1));
                masks.add(manager.create(1).reshape(1));
                labels.add(manager.create(0).reshape(1));
                count += 1;
            }
            // Fill with zeroes remaining array
            while (count != maxLen) {
                contextsNegatives.add(manager.create(0).reshape(1));
                masks.add(manager.create(0).reshape(1));
                labels.add(manager.create(0).reshape(1));
                count += 1;
            }

            // Add this NDArrays to output NDArrays
            centers.add(center.reshape(1));
        }
        return new NDList(
                NDArrays.concat(centers).reshape(data.length, -1),
                NDArrays.concat(contextsNegatives).reshape(data.length, -1),
                NDArrays.concat(masks).reshape(data.length, -1),
                NDArrays.concat(labels).reshape(data.length, -1));
    }

    public static NDList convertNDArray(Object[] data, NDManager manager) {
        ArrayList<Integer> centers = (ArrayList<Integer>) data[0];
        ArrayList<ArrayList<Integer>> contexts = (ArrayList<ArrayList<Integer>>) data[1];
        ArrayList<ArrayList<Integer>> negatives = (ArrayList<ArrayList<Integer>>) data[2];

        // Create centers NDArray
        NDArray centersNDArray = manager.create(centers.stream().mapToInt(i -> i).toArray());

        // Create contexts NDArray
        int maxLen = 0;
        for (ArrayList<Integer> context : contexts) {
            maxLen = Math.max(maxLen, context.size());
        }
        // Fill arrays with 0s to all have same lengths and be able to create NDArray
        for (ArrayList<Integer> context : contexts) {
            while (context.size() != maxLen) {
                context.add(0);
            }
        }
        NDArray contextsNDArray =
                manager.create(
                        contexts.stream()
                                .map(u -> u.stream().mapToInt(i -> i).toArray())
                                .toArray(int[][]::new));

        // Create negatives NDArray
        maxLen = 0;
        for (ArrayList<Integer> negative : negatives) {
            maxLen = Math.max(maxLen, negative.size());
        }
        // Fill arrays with 0s to all have same lengths and be able to create NDArray
        for (ArrayList<Integer> negative : negatives) {
            while (negative.size() != maxLen) {
                negative.add(0);
            }
        }
        NDArray negativesNDArray =
                manager.create(
                        negatives.stream()
                                .map(u -> u.stream().mapToInt(i -> i).toArray())
                                .toArray(int[][]::new));

        return new NDList(centersNDArray, contextsNDArray, negativesNDArray);
    }

    public static Pair<ArrayDataset, Vocab> loadDataPTB(

            int batchSize, int maxWindowSize, int numNoiseWords, NDManager manager)
            throws IOException, TranslateException {
        String[][] sentences = readPTB();
        Vocab vocab = new Vocab(sentences, 10, new String[]{});
        String[][] subSampled = subSampling(sentences, vocab);
        Integer[][] corpus = new Integer[subSampled.length][];
        for (int i = 0; i < subSampled.length; i++) {
            corpus[i] = vocab.getIdxs(subSampled[i]);
        }
        Pair<ArrayList<Integer>, ArrayList<ArrayList<Integer>>> pair =
                getCentersAndContext(corpus, maxWindowSize);
        ArrayList<ArrayList<Integer>> negatives =
                getNegatives(pair.getValue(), corpus, numNoiseWords);

        NDList ndArrays =
                convertNDArray(new Object[]{pair.getKey(), pair.getValue(), negatives}, manager);
        ArrayDataset dataset =
                new ArrayDataset.Builder()
                        .setData(ndArrays.get(0), ndArrays.get(1), ndArrays.get(2))
                        .optDataBatchifier(
                                new Batchifier() {
                                    @Override
                                    public NDList batchify(NDList[] ndLists) {
                                        return batchifyData(manager, ndLists);
                                    }

                                    @Override
                                    public NDList[] unbatchify(NDList ndList) {
                                        return new NDList[0];
                                    }
                                })
                        .setSampling(batchSize, true)
                        .build();

        return new Pair<>(dataset, vocab);
    }
}
