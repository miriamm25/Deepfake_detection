---
license: cc-by-sa-4.0
extra_gated_prompt: "You agree to not use this dataset for the development or improvement of any technologies which have the potential to harm individuals, institutions, or societies."
extra_gated_fields:
  Full name: text
  Institution or Company: text
  Insitutional email address: text
  Country: country
  Specific date: date_picker
  I want to use this model to: text
  Link to a verifiable source with evidence (ex website, paper, or news releases) that you have done work related to deepfake detection or a related field such as computer vision or AI alignment research in the past: text
---

# Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark of Deepfakes Circulated in 2024 
Deepfake-Eval-2024 is an in-the-wild deepfake dataset. Deepfake-Eval-2024 contains 44 hours of videos, 56.5 hours of audio, and 1,975 images, encompassing contemporary manipulation technologies, diverse media content, 88 different website sources, and 52 different languages. Deepfake-Eval-2024 contains manually labeled real and fake media. Deepfake-Eval-2024 is designed to facilitate deepfake detection research. Deepfake-Eval-2024 was created by a team from TrueMedia.org, the University of Washington, Miraflow AI, Georgetown University, Chung-Ang University, and Yonsei University. 

![Examples of Deepfake-Eval-2024 video and audio (rows 1–2), and images (rows 3–4),
demonstrating a diversity of content styles and generation techniques, including lipsync, faceswap, and diffusion. Images have been resized for presentation.](examples/fig1_examples.png)

WARNING: There was an error in the video metadata file. It has been corrected 10-29-2025

## Dataset Details 
- **[Paper](https://arxiv.org/abs/2503.02857)**
- **[Repository](https://github.com/nuriachandra/Deepfake-Eval-2024)**


## Uses

### Direct Use
Deepfake-Eval-2024 is designed to facilitate deepfake detection research. The dataset can be used to benchmark detection methods. Deepfake-Eval-2024 also provides a large sample of the deepfakes circulating in 2024 and can be used for social science research characterizing deepfake use. 

### Out-of-Scope Use
Deepfake-Eval-2024 was built to evaluate deepfake **detection** methods. Use of Deepfake-Eval-2024 to train generative AI models is out-of-scope. 



## Datset Creation

### Dataset Creation Rational
Deepfake-Eval-2024 was created in order to evaluate deepfake detection models on contemporary, real-world data, that is circulated on social media, and other content-sharing platforms. Existing academic datasets were largely outdated, and lacked diversity in media content and style.  

### Data Collection and Processing
**Data collection** Data was collected through the deepfake detection platform TrueMedia.org, and X (formerly Twitter) via the following mechanisms: 
1. A TrueMedia.org user copied a website or social media link into TrueMedia.org.
2. A TrueMedia.org user directly uploaded media to TrueMedia.org.
3. An X user tagged the TrueMedia.org bot to upload the media to TrueMedia.org.
4. An X Community Notes member flagged an X post as potentially AI-manipulated. Flagged posts were scrapped and added to our dataset. 

**Data Labeling**: Please refer to the Deepfake-Eval-2024 paper for details on the data labeling process and criteria. 

**Data filtering**:
1. We remove duplicate data. However, we include cases where two pieces of media have minor, non-visible variations, and thus appear to be the same (e.g., different cropping of the same video). 
2. In order to tailor our datasets to evaluate deepfake detection models, we remove images and videos that do not contain photorealistic faces.
3. We remove data that labelers were unable to categorize. Labeled videos are still included in the Deepfake-Eval-2024-video dataset even when the label of their corresponding audio could not be
determined. Likewise, labeled audio from videos is included in Deepfake-Eval-2024-audio even when the corresponding video is unlabeled.




## Personal and Sensitive Information
The datasets are comprised of media publicly available on social media websites, news websites, and media uploaded to TrueMedia.org to be checked for AI manipulation. We are not aware of any sensitive personal identification information included in the dataset. However, we cannot guarantee that media uploaded from users or from web sources does not contain any personal or sensitive information that the individual in question did not consent to share. We note that TrueMedia.org users were informed in the Terms of Use that 'anything you share will not be private and can be used by us, our partners, and others we work with, in lots of different ways.'

NSFW content was explicitly forbidden in the TrueMedia.org Terms of Use, and users uploading pornographic content were blocked from using the application, and their uploaded data was deleted. However, **no explicit post-hoc filtering to remove NSFW content from the dataset was applied**. As such, we caution that our dataset contains a small quantity of NSFW content. 


## Bias, Risks, and Limitations

**Bias**: Although Deepfake-Eval-2024 is significantly more diverse than existing deepfake datasets, the majority of the dataset is still comprised of English-speaking, light-skinned individuals from Western countries, with a particularly high number from America. 

**Potential Legal and Ethical Concerns**: Deepfakes pose a threat to society and privacy. We note that this dataset could be used to train increasingly realistic deepfakes, and potentially design deepfakes that will evade detection models that are evaluated on Deepfake-Eval-2024. As such, we gate access to individuals and organizations who are conducting work related to deepfake detection. 


## License
We use a CC BY-SA 4.0 license, which allows for commercial use so that this work can foster the development of improved deepfake detection models in both the commercial and academic sectors. 


## Citation
```
@misc{chandra2025deepfakeeval2024multimodalinthewildbenchmark,
      title={Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark of Deepfakes Circulated in 2024}, 
      author={Nuria Alina Chandra and Ryan Murtfeldt and Lin Qiu and Arnab Karmakar and Hannah Lee and Emmanuel Tanumihardja and Kevin Farhat and Ben Caffee and Sejin Paik and Changyeon Lee and Jongwook Choi and Aerin Kim and Oren Etzioni},
      year={2025},
      eprint={2503.02857},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.02857}, 
}
```