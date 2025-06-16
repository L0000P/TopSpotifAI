# TopSpotifAI

**TopSpotifAI** is a machine learning project that analyzes **Top 50 daily Spotify playlists by country** to predict the **future position** and **popularity** of songs over the following days using models such as:

- **Ridge Regressor**
- **Enhanced MLP** (inspired by Time Series Transformers)

## Dataset

The dataset used for training and evaluation is available on Kaggle:

🔗 [Spotify Top 50 Playlist Songs](https://www.kaggle.com/datasets/anxods/spotify-top-50-playlist-songs-anxods)

Ensure the dataset is placed in the following directory structure:
```bash
TopSpotifAI/
└── src/
  └── data/
    └── *.csv
```
## Docker

The entire project is containerized for easy setup and reproducibility.

***NOTE***: Modify **shm_size** parameter in **docker-compose.yml** files for your GPU memory support, in this case 8GB: 

```yaml
shm_size: "8gb"
```

### Dev Build

```console
$: ./TopSpotifAI dev up --build
```
In this mode inside the cointainer, you must install libs:
```console
$: pip install -r requirements.txt
```
and test your desired model:
```console
$: python /TopSpotifAI/models/RidgeRegressor/test.py
$: python /TopSpotifAI/models/EnhancedMLP/test.py
```

### Prod Build 

```console
$: ./TopSpotifAI prod up --build
```
In this mode, the container install the libs and train all models (if not trained).
