package com.hyfly.unit12;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.MultiBoxPrior;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import com.hyfly.utils.ImageUtils;
import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;
import java.util.List;

@Slf4j
public class Test03 {

    public static void main(String[] args) throws Exception {
        Image img = ImageFactory.getInstance()
                .fromUrl("https://raw.githubusercontent.com/d2l-ai/d2l-en/master/img/catdog.jpg");
        int HEIGHT = img.getHeight();
        int WIDTH = img.getWidth();

        System.out.printf("%d, %d\n%n", HEIGHT, WIDTH);

        //
        Image img2 = img.duplicate();
        displayAnchors(img2, 4, 4, Arrays.asList(0.15f));

        img2.getWrappedImage();

        Image img3 = img.duplicate();
        displayAnchors(img3, 2, 2, Arrays.asList(0.4f));

        img3.getWrappedImage();

        Image img4 = img.duplicate();
        displayAnchors(img4, 1, 1, Arrays.asList(0.8f));

        img4.getWrappedImage();
    }

    public static void displayAnchors(Image img, int fmapWidth, int fmapHeight, List<Float> sizes) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray fmap = manager.zeros(new Shape(1, 10, fmapWidth, fmapHeight));

            List<Float> ratios = Arrays.asList(1f, 2f, 0.5f);

            MultiBoxPrior mbp = MultiBoxPrior.builder().setSizes(sizes).setRatios(ratios).build();
            NDArray anchors = mbp.generateAnchorBoxes(fmap);

            ImageUtils.drawBBoxes(img, anchors.get(0), null);
        }

    }
}
