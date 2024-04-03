package com.hyfly.unit07;

import ai.djl.util.Pair;
import com.hyfly.utils.timemachine.TimeMachine;
import com.hyfly.utils.timemachine.Vocab;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;


public class Test02 {

    public static void main(String[] args) throws Exception {
        String[] lines = readTimeMachine();
        System.out.println("# text lines: " + lines.length);
        System.out.println(lines[0]);
        System.out.println(lines[10]);

        String[][] tokens = tokenize(lines, "word");
        for (int i = 0; i < 11; i++) {
            System.out.println(Arrays.toString(tokens[i]));
        }

        Vocab vocab = new Vocab(tokens, 0, new String[0]);
        for (int i = 0; i < 10; i++) {
            String token = vocab.idxToToken.get(i);
            System.out.print("(" + token + ", " + vocab.tokenToIdx.get(token) + ") ");
        }

        for (int i : new int[]{0, 10}) {
            System.out.println("Words:" + Arrays.toString(tokens[i]));
            System.out.println("Indices:" + Arrays.toString(vocab.getIdxs(tokens[i])));
        }

        Pair<List<Integer>, Vocab> corpusVocabPair = TimeMachine.loadCorpusTimeMachine(-1);
        List<Integer> corpus = corpusVocabPair.getKey();
        vocab = corpusVocabPair.getValue();

        System.out.println(corpus.size());
        System.out.println(vocab.length());
    }

    public static String[] readTimeMachine() throws IOException {
        URL url = new URL("http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt");
        String[] lines;
        try (BufferedReader in = new BufferedReader(new InputStreamReader(url.openStream()))) {
            lines = in.lines().toArray(String[]::new);
        }

        for (int i = 0; i < lines.length; i++) {
            lines[i] = lines[i].replaceAll("[^A-Za-z]+", " ").strip().toLowerCase();
        }
        return lines;
    }

    public static String[][] tokenize(String[] lines, String token) throws Exception {
        // 将文本行拆分为单词或字符标记
        String[][] output = new String[lines.length][];
        if (token == "word") {
            for (int i = 0; i < output.length; i++) {
                output[i] = lines[i].split(" ");
            }
        } else if (token == "char") {
            for (int i = 0; i < output.length; i++) {
                output[i] = lines[i].split("");
            }
        } else {
            throw new Exception("ERROR: unknown token type: " + token);
        }
        return output;
    }

    public static LinkedHashMap<String, Integer> countCorpus(String[] tokens) {
        /* 计算token频率 */
        LinkedHashMap<String, Integer> counter = new LinkedHashMap<>();
        if (tokens.length != 0) {
            for (String token : tokens) {
                counter.put(token, counter.getOrDefault(token, 0) + 1);
            }
        }
        return counter;
    }

    public static LinkedHashMap<String, Integer> countCorpus2D(String[][] tokens) {
        /* 将token列表展平为token列表*/
        List<String> allTokens = new ArrayList<String>();
        for (int i = 0; i < tokens.length; i++) {
            for (int j = 0; j < tokens[i].length; j++) {
                if (tokens[i][j] != "") {
                    allTokens.add(tokens[i][j]);
                }
            }
        }
        return countCorpus(allTokens.toArray(new String[0]));
    }
}
