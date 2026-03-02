{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "toc_visible": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "N4lRFW9KBExJ"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import sklearn.datasets\n",
        "from sklearn.model_selection import train_test_split\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# loading the data from sklearn\n",
        "breast_cancer_dataset = sklearn.datasets.load_breast_cancer()\n"
      ],
      "metadata": {
        "id": "H4Jl8BxCCtpQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(breast_cancer_dataset)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fsrFF6lEF-6T",
        "outputId": "37a7eaf8-f974-4cca-fed8-80ffab370986"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{'data': array([[1.799e+01, 1.038e+01, 1.228e+02, ..., 2.654e-01, 4.601e-01,\n",
            "        1.189e-01],\n",
            "       [2.057e+01, 1.777e+01, 1.329e+02, ..., 1.860e-01, 2.750e-01,\n",
            "        8.902e-02],\n",
            "       [1.969e+01, 2.125e+01, 1.300e+02, ..., 2.430e-01, 3.613e-01,\n",
            "        8.758e-02],\n",
            "       ...,\n",
            "       [1.660e+01, 2.808e+01, 1.083e+02, ..., 1.418e-01, 2.218e-01,\n",
            "        7.820e-02],\n",
            "       [2.060e+01, 2.933e+01, 1.401e+02, ..., 2.650e-01, 4.087e-01,\n",
            "        1.240e-01],\n",
            "       [7.760e+00, 2.454e+01, 4.792e+01, ..., 0.000e+00, 2.871e-01,\n",
            "        7.039e-02]]), 'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,\n",
            "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,\n",
            "       0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,\n",
            "       1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,\n",
            "       1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,\n",
            "       1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0,\n",
            "       0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,\n",
            "       1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,\n",
            "       1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0,\n",
            "       0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,\n",
            "       1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1,\n",
            "       1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "       0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,\n",
            "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,\n",
            "       1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0,\n",
            "       0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,\n",
            "       0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,\n",
            "       1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,\n",
            "       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,\n",
            "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1,\n",
            "       1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,\n",
            "       1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,\n",
            "       1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,\n",
            "       1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,\n",
            "       1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
            "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]), 'frame': None, 'target_names': array(['malignant', 'benign'], dtype='<U9'), 'DESCR': '.. _breast_cancer_dataset:\\n\\nBreast cancer wisconsin (diagnostic) dataset\\n--------------------------------------------\\n\\n**Data Set Characteristics:**\\n\\n:Number of Instances: 569\\n\\n:Number of Attributes: 30 numeric, predictive attributes and the class\\n\\n:Attribute Information:\\n    - radius (mean of distances from center to points on the perimeter)\\n    - texture (standard deviation of gray-scale values)\\n    - perimeter\\n    - area\\n    - smoothness (local variation in radius lengths)\\n    - compactness (perimeter^2 / area - 1.0)\\n    - concavity (severity of concave portions of the contour)\\n    - concave points (number of concave portions of the contour)\\n    - symmetry\\n    - fractal dimension (\"coastline approximation\" - 1)\\n\\n    The mean, standard error, and \"worst\" or largest (mean of the three\\n    worst/largest values) of these features were computed for each image,\\n    resulting in 30 features.  For instance, field 0 is Mean Radius, field\\n    10 is Radius SE, field 20 is Worst Radius.\\n\\n    - class:\\n            - WDBC-Malignant\\n            - WDBC-Benign\\n\\n:Summary Statistics:\\n\\n===================================== ====== ======\\n                                        Min    Max\\n===================================== ====== ======\\nradius (mean):                        6.981  28.11\\ntexture (mean):                       9.71   39.28\\nperimeter (mean):                     43.79  188.5\\narea (mean):                          143.5  2501.0\\nsmoothness (mean):                    0.053  0.163\\ncompactness (mean):                   0.019  0.345\\nconcavity (mean):                     0.0    0.427\\nconcave points (mean):                0.0    0.201\\nsymmetry (mean):                      0.106  0.304\\nfractal dimension (mean):             0.05   0.097\\nradius (standard error):              0.112  2.873\\ntexture (standard error):             0.36   4.885\\nperimeter (standard error):           0.757  21.98\\narea (standard error):                6.802  542.2\\nsmoothness (standard error):          0.002  0.031\\ncompactness (standard error):         0.002  0.135\\nconcavity (standard error):           0.0    0.396\\nconcave points (standard error):      0.0    0.053\\nsymmetry (standard error):            0.008  0.079\\nfractal dimension (standard error):   0.001  0.03\\nradius (worst):                       7.93   36.04\\ntexture (worst):                      12.02  49.54\\nperimeter (worst):                    50.41  251.2\\narea (worst):                         185.2  4254.0\\nsmoothness (worst):                   0.071  0.223\\ncompactness (worst):                  0.027  1.058\\nconcavity (worst):                    0.0    1.252\\nconcave points (worst):               0.0    0.291\\nsymmetry (worst):                     0.156  0.664\\nfractal dimension (worst):            0.055  0.208\\n===================================== ====== ======\\n\\n:Missing Attribute Values: None\\n\\n:Class Distribution: 212 - Malignant, 357 - Benign\\n\\n:Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian\\n\\n:Donor: Nick Street\\n\\n:Date: November, 1995\\n\\nThis is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.\\nhttps://goo.gl/U2Uwz2\\n\\nFeatures are computed from a digitized image of a fine needle\\naspirate (FNA) of a breast mass.  They describe\\ncharacteristics of the cell nuclei present in the image.\\n\\nSeparating plane described above was obtained using\\nMultisurface Method-Tree (MSM-T) [K. P. Bennett, \"Decision Tree\\nConstruction Via Linear Programming.\" Proceedings of the 4th\\nMidwest Artificial Intelligence and Cognitive Science Society,\\npp. 97-101, 1992], a classification method which uses linear\\nprogramming to construct a decision tree.  Relevant features\\nwere selected using an exhaustive search in the space of 1-4\\nfeatures and 1-3 separating planes.\\n\\nThe actual linear program used to obtain the separating plane\\nin the 3-dimensional space is that described in:\\n[K. P. Bennett and O. L. Mangasarian: \"Robust Linear\\nProgramming Discrimination of Two Linearly Inseparable Sets\",\\nOptimization Methods and Software 1, 1992, 23-34].\\n\\nThis database is also available through the UW CS ftp server:\\n\\nftp ftp.cs.wisc.edu\\ncd math-prog/cpo-dataset/machine-learn/WDBC/\\n\\n.. dropdown:: References\\n\\n  - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction\\n    for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on\\n    Electronic Imaging: Science and Technology, volume 1905, pages 861-870,\\n    San Jose, CA, 1993.\\n  - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and\\n    prognosis via linear programming. Operations Research, 43(4), pages 570-577,\\n    July-August 1995.\\n  - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques\\n    to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994)\\n    163-171.\\n', 'feature_names': array(['mean radius', 'mean texture', 'mean perimeter', 'mean area',\n",
            "       'mean smoothness', 'mean compactness', 'mean concavity',\n",
            "       'mean concave points', 'mean symmetry', 'mean fractal dimension',\n",
            "       'radius error', 'texture error', 'perimeter error', 'area error',\n",
            "       'smoothness error', 'compactness error', 'concavity error',\n",
            "       'concave points error', 'symmetry error',\n",
            "       'fractal dimension error', 'worst radius', 'worst texture',\n",
            "       'worst perimeter', 'worst area', 'worst smoothness',\n",
            "       'worst compactness', 'worst concavity', 'worst concave points',\n",
            "       'worst symmetry', 'worst fractal dimension'], dtype='<U23'), 'filename': 'breast_cancer.csv', 'data_module': 'sklearn.datasets.data'}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# loading the data to a data frame\n",
        "data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)"
      ],
      "metadata": {
        "id": "50OyQULaGHVt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# print the first 5 rows of the dataframe\n",
        "data_frame.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 290
        },
        "id": "bZdEwHAxGOwh",
        "outputId": "6ed87b2b-a464-4c29-c9ca-267ee676d246"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   mean radius  mean texture  mean perimeter  mean area  mean smoothness  \\\n",
              "0        17.99         10.38          122.80     1001.0          0.11840   \n",
              "1        20.57         17.77          132.90     1326.0          0.08474   \n",
              "2        19.69         21.25          130.00     1203.0          0.10960   \n",
              "3        11.42         20.38           77.58      386.1          0.14250   \n",
              "4        20.29         14.34          135.10     1297.0          0.10030   \n",
              "\n",
              "   mean compactness  mean concavity  mean concave points  mean symmetry  \\\n",
              "0           0.27760          0.3001              0.14710         0.2419   \n",
              "1           0.07864          0.0869              0.07017         0.1812   \n",
              "2           0.15990          0.1974              0.12790         0.2069   \n",
              "3           0.28390          0.2414              0.10520         0.2597   \n",
              "4           0.13280          0.1980              0.10430         0.1809   \n",
              "\n",
              "   mean fractal dimension  ...  worst radius  worst texture  worst perimeter  \\\n",
              "0                 0.07871  ...         25.38          17.33           184.60   \n",
              "1                 0.05667  ...         24.99          23.41           158.80   \n",
              "2                 0.05999  ...         23.57          25.53           152.50   \n",
              "3                 0.09744  ...         14.91          26.50            98.87   \n",
              "4                 0.05883  ...         22.54          16.67           152.20   \n",
              "\n",
              "   worst area  worst smoothness  worst compactness  worst concavity  \\\n",
              "0      2019.0            0.1622             0.6656           0.7119   \n",
              "1      1956.0            0.1238             0.1866           0.2416   \n",
              "2      1709.0            0.1444             0.4245           0.4504   \n",
              "3       567.7            0.2098             0.8663           0.6869   \n",
              "4      1575.0            0.1374             0.2050           0.4000   \n",
              "\n",
              "   worst concave points  worst symmetry  worst fractal dimension  \n",
              "0                0.2654          0.4601                  0.11890  \n",
              "1                0.1860          0.2750                  0.08902  \n",
              "2                0.2430          0.3613                  0.08758  \n",
              "3                0.2575          0.6638                  0.17300  \n",
              "4                0.1625          0.2364                  0.07678  \n",
              "\n",
              "[5 rows x 30 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-fb7087fc-1b46-4e1c-ad9b-df0c85a25754\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>mean radius</th>\n",
              "      <th>mean texture</th>\n",
              "      <th>mean perimeter</th>\n",
              "      <th>mean area</th>\n",
              "      <th>mean smoothness</th>\n",
              "      <th>mean compactness</th>\n",
              "      <th>mean concavity</th>\n",
              "      <th>mean concave points</th>\n",
              "      <th>mean symmetry</th>\n",
              "      <th>mean fractal dimension</th>\n",
              "      <th>...</th>\n",
              "      <th>worst radius</th>\n",
              "      <th>worst texture</th>\n",
              "      <th>worst perimeter</th>\n",
              "      <th>worst area</th>\n",
              "      <th>worst smoothness</th>\n",
              "      <th>worst compactness</th>\n",
              "      <th>worst concavity</th>\n",
              "      <th>worst concave points</th>\n",
              "      <th>worst symmetry</th>\n",
              "      <th>worst fractal dimension</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>17.99</td>\n",
              "      <td>10.38</td>\n",
              "      <td>122.80</td>\n",
              "      <td>1001.0</td>\n",
              "      <td>0.11840</td>\n",
              "      <td>0.27760</td>\n",
              "      <td>0.3001</td>\n",
              "      <td>0.14710</td>\n",
              "      <td>0.2419</td>\n",
              "      <td>0.07871</td>\n",
              "      <td>...</td>\n",
              "      <td>25.38</td>\n",
              "      <td>17.33</td>\n",
              "      <td>184.60</td>\n",
              "      <td>2019.0</td>\n",
              "      <td>0.1622</td>\n",
              "      <td>0.6656</td>\n",
              "      <td>0.7119</td>\n",
              "      <td>0.2654</td>\n",
              "      <td>0.4601</td>\n",
              "      <td>0.11890</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>20.57</td>\n",
              "      <td>17.77</td>\n",
              "      <td>132.90</td>\n",
              "      <td>1326.0</td>\n",
              "      <td>0.08474</td>\n",
              "      <td>0.07864</td>\n",
              "      <td>0.0869</td>\n",
              "      <td>0.07017</td>\n",
              "      <td>0.1812</td>\n",
              "      <td>0.05667</td>\n",
              "      <td>...</td>\n",
              "      <td>24.99</td>\n",
              "      <td>23.41</td>\n",
              "      <td>158.80</td>\n",
              "      <td>1956.0</td>\n",
              "      <td>0.1238</td>\n",
              "      <td>0.1866</td>\n",
              "      <td>0.2416</td>\n",
              "      <td>0.1860</td>\n",
              "      <td>0.2750</td>\n",
              "      <td>0.08902</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>19.69</td>\n",
              "      <td>21.25</td>\n",
              "      <td>130.00</td>\n",
              "      <td>1203.0</td>\n",
              "      <td>0.10960</td>\n",
              "      <td>0.15990</td>\n",
              "      <td>0.1974</td>\n",
              "      <td>0.12790</td>\n",
              "      <td>0.2069</td>\n",
              "      <td>0.05999</td>\n",
              "      <td>...</td>\n",
              "      <td>23.57</td>\n",
              "      <td>25.53</td>\n",
              "      <td>152.50</td>\n",
              "      <td>1709.0</td>\n",
              "      <td>0.1444</td>\n",
              "      <td>0.4245</td>\n",
              "      <td>0.4504</td>\n",
              "      <td>0.2430</td>\n",
              "      <td>0.3613</td>\n",
              "      <td>0.08758</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>11.42</td>\n",
              "      <td>20.38</td>\n",
              "      <td>77.58</td>\n",
              "      <td>386.1</td>\n",
              "      <td>0.14250</td>\n",
              "      <td>0.28390</td>\n",
              "      <td>0.2414</td>\n",
              "      <td>0.10520</td>\n",
              "      <td>0.2597</td>\n",
              "      <td>0.09744</td>\n",
              "      <td>...</td>\n",
              "      <td>14.91</td>\n",
              "      <td>26.50</td>\n",
              "      <td>98.87</td>\n",
              "      <td>567.7</td>\n",
              "      <td>0.2098</td>\n",
              "      <td>0.8663</td>\n",
              "      <td>0.6869</td>\n",
              "      <td>0.2575</td>\n",
              "      <td>0.6638</td>\n",
              "      <td>0.17300</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>20.29</td>\n",
              "      <td>14.34</td>\n",
              "      <td>135.10</td>\n",
              "      <td>1297.0</td>\n",
              "      <td>0.10030</td>\n",
              "      <td>0.13280</td>\n",
              "      <td>0.1980</td>\n",
              "      <td>0.10430</td>\n",
              "      <td>0.1809</td>\n",
              "      <td>0.05883</td>\n",
              "      <td>...</td>\n",
              "      <td>22.54</td>\n",
              "      <td>16.67</td>\n",
              "      <td>152.20</td>\n",
              "      <td>1575.0</td>\n",
              "      <td>0.1374</td>\n",
              "      <td>0.2050</td>\n",
              "      <td>0.4000</td>\n",
              "      <td>0.1625</td>\n",
              "      <td>0.2364</td>\n",
              "      <td>0.07678</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 30 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-fb7087fc-1b46-4e1c-ad9b-df0c85a25754')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-fb7087fc-1b46-4e1c-ad9b-df0c85a25754 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-fb7087fc-1b46-4e1c-ad9b-df0c85a25754');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "data_frame"
            }
          },
          "metadata": {},
          "execution_count": 59
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# adding the 'target' column to the data frame\n",
        "data_frame['label'] = breast_cancer_dataset.target"
      ],
      "metadata": {
        "id": "p4-kni5-GZMD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# print last 5 rows of the dataframe\n",
        "data_frame.tail()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 290
        },
        "id": "tvxcbJHWGe1g",
        "outputId": "6b7a90c7-92e7-4c98-c599-a7a2ac3fa01d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "     mean radius  mean texture  mean perimeter  mean area  mean smoothness  \\\n",
              "564        21.56         22.39          142.00     1479.0          0.11100   \n",
              "565        20.13         28.25          131.20     1261.0          0.09780   \n",
              "566        16.60         28.08          108.30      858.1          0.08455   \n",
              "567        20.60         29.33          140.10     1265.0          0.11780   \n",
              "568         7.76         24.54           47.92      181.0          0.05263   \n",
              "\n",
              "     mean compactness  mean concavity  mean concave points  mean symmetry  \\\n",
              "564           0.11590         0.24390              0.13890         0.1726   \n",
              "565           0.10340         0.14400              0.09791         0.1752   \n",
              "566           0.10230         0.09251              0.05302         0.1590   \n",
              "567           0.27700         0.35140              0.15200         0.2397   \n",
              "568           0.04362         0.00000              0.00000         0.1587   \n",
              "\n",
              "     mean fractal dimension  ...  worst texture  worst perimeter  worst area  \\\n",
              "564                 0.05623  ...          26.40           166.10      2027.0   \n",
              "565                 0.05533  ...          38.25           155.00      1731.0   \n",
              "566                 0.05648  ...          34.12           126.70      1124.0   \n",
              "567                 0.07016  ...          39.42           184.60      1821.0   \n",
              "568                 0.05884  ...          30.37            59.16       268.6   \n",
              "\n",
              "     worst smoothness  worst compactness  worst concavity  \\\n",
              "564           0.14100            0.21130           0.4107   \n",
              "565           0.11660            0.19220           0.3215   \n",
              "566           0.11390            0.30940           0.3403   \n",
              "567           0.16500            0.86810           0.9387   \n",
              "568           0.08996            0.06444           0.0000   \n",
              "\n",
              "     worst concave points  worst symmetry  worst fractal dimension  label  \n",
              "564                0.2216          0.2060                  0.07115      0  \n",
              "565                0.1628          0.2572                  0.06637      0  \n",
              "566                0.1418          0.2218                  0.07820      0  \n",
              "567                0.2650          0.4087                  0.12400      0  \n",
              "568                0.0000          0.2871                  0.07039      1  \n",
              "\n",
              "[5 rows x 31 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-e6fae56f-dd8a-455c-b8f1-005edc9cc803\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>mean radius</th>\n",
              "      <th>mean texture</th>\n",
              "      <th>mean perimeter</th>\n",
              "      <th>mean area</th>\n",
              "      <th>mean smoothness</th>\n",
              "      <th>mean compactness</th>\n",
              "      <th>mean concavity</th>\n",
              "      <th>mean concave points</th>\n",
              "      <th>mean symmetry</th>\n",
              "      <th>mean fractal dimension</th>\n",
              "      <th>...</th>\n",
              "      <th>worst texture</th>\n",
              "      <th>worst perimeter</th>\n",
              "      <th>worst area</th>\n",
              "      <th>worst smoothness</th>\n",
              "      <th>worst compactness</th>\n",
              "      <th>worst concavity</th>\n",
              "      <th>worst concave points</th>\n",
              "      <th>worst symmetry</th>\n",
              "      <th>worst fractal dimension</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>564</th>\n",
              "      <td>21.56</td>\n",
              "      <td>22.39</td>\n",
              "      <td>142.00</td>\n",
              "      <td>1479.0</td>\n",
              "      <td>0.11100</td>\n",
              "      <td>0.11590</td>\n",
              "      <td>0.24390</td>\n",
              "      <td>0.13890</td>\n",
              "      <td>0.1726</td>\n",
              "      <td>0.05623</td>\n",
              "      <td>...</td>\n",
              "      <td>26.40</td>\n",
              "      <td>166.10</td>\n",
              "      <td>2027.0</td>\n",
              "      <td>0.14100</td>\n",
              "      <td>0.21130</td>\n",
              "      <td>0.4107</td>\n",
              "      <td>0.2216</td>\n",
              "      <td>0.2060</td>\n",
              "      <td>0.07115</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>565</th>\n",
              "      <td>20.13</td>\n",
              "      <td>28.25</td>\n",
              "      <td>131.20</td>\n",
              "      <td>1261.0</td>\n",
              "      <td>0.09780</td>\n",
              "      <td>0.10340</td>\n",
              "      <td>0.14400</td>\n",
              "      <td>0.09791</td>\n",
              "      <td>0.1752</td>\n",
              "      <td>0.05533</td>\n",
              "      <td>...</td>\n",
              "      <td>38.25</td>\n",
              "      <td>155.00</td>\n",
              "      <td>1731.0</td>\n",
              "      <td>0.11660</td>\n",
              "      <td>0.19220</td>\n",
              "      <td>0.3215</td>\n",
              "      <td>0.1628</td>\n",
              "      <td>0.2572</td>\n",
              "      <td>0.06637</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>566</th>\n",
              "      <td>16.60</td>\n",
              "      <td>28.08</td>\n",
              "      <td>108.30</td>\n",
              "      <td>858.1</td>\n",
              "      <td>0.08455</td>\n",
              "      <td>0.10230</td>\n",
              "      <td>0.09251</td>\n",
              "      <td>0.05302</td>\n",
              "      <td>0.1590</td>\n",
              "      <td>0.05648</td>\n",
              "      <td>...</td>\n",
              "      <td>34.12</td>\n",
              "      <td>126.70</td>\n",
              "      <td>1124.0</td>\n",
              "      <td>0.11390</td>\n",
              "      <td>0.30940</td>\n",
              "      <td>0.3403</td>\n",
              "      <td>0.1418</td>\n",
              "      <td>0.2218</td>\n",
              "      <td>0.07820</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>567</th>\n",
              "      <td>20.60</td>\n",
              "      <td>29.33</td>\n",
              "      <td>140.10</td>\n",
              "      <td>1265.0</td>\n",
              "      <td>0.11780</td>\n",
              "      <td>0.27700</td>\n",
              "      <td>0.35140</td>\n",
              "      <td>0.15200</td>\n",
              "      <td>0.2397</td>\n",
              "      <td>0.07016</td>\n",
              "      <td>...</td>\n",
              "      <td>39.42</td>\n",
              "      <td>184.60</td>\n",
              "      <td>1821.0</td>\n",
              "      <td>0.16500</td>\n",
              "      <td>0.86810</td>\n",
              "      <td>0.9387</td>\n",
              "      <td>0.2650</td>\n",
              "      <td>0.4087</td>\n",
              "      <td>0.12400</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>568</th>\n",
              "      <td>7.76</td>\n",
              "      <td>24.54</td>\n",
              "      <td>47.92</td>\n",
              "      <td>181.0</td>\n",
              "      <td>0.05263</td>\n",
              "      <td>0.04362</td>\n",
              "      <td>0.00000</td>\n",
              "      <td>0.00000</td>\n",
              "      <td>0.1587</td>\n",
              "      <td>0.05884</td>\n",
              "      <td>...</td>\n",
              "      <td>30.37</td>\n",
              "      <td>59.16</td>\n",
              "      <td>268.6</td>\n",
              "      <td>0.08996</td>\n",
              "      <td>0.06444</td>\n",
              "      <td>0.0000</td>\n",
              "      <td>0.0000</td>\n",
              "      <td>0.2871</td>\n",
              "      <td>0.07039</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 31 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-e6fae56f-dd8a-455c-b8f1-005edc9cc803')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-e6fae56f-dd8a-455c-b8f1-005edc9cc803 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-e6fae56f-dd8a-455c-b8f1-005edc9cc803');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe"
            }
          },
          "metadata": {},
          "execution_count": 61
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# getting some information about the data\n",
        "data_frame.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pJyqQg1zGjxP",
        "outputId": "269bc720-2a8a-4a18-9d47-30a64f098c4b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 569 entries, 0 to 568\n",
            "Data columns (total 31 columns):\n",
            " #   Column                   Non-Null Count  Dtype  \n",
            "---  ------                   --------------  -----  \n",
            " 0   mean radius              569 non-null    float64\n",
            " 1   mean texture             569 non-null    float64\n",
            " 2   mean perimeter           569 non-null    float64\n",
            " 3   mean area                569 non-null    float64\n",
            " 4   mean smoothness          569 non-null    float64\n",
            " 5   mean compactness         569 non-null    float64\n",
            " 6   mean concavity           569 non-null    float64\n",
            " 7   mean concave points      569 non-null    float64\n",
            " 8   mean symmetry            569 non-null    float64\n",
            " 9   mean fractal dimension   569 non-null    float64\n",
            " 10  radius error             569 non-null    float64\n",
            " 11  texture error            569 non-null    float64\n",
            " 12  perimeter error          569 non-null    float64\n",
            " 13  area error               569 non-null    float64\n",
            " 14  smoothness error         569 non-null    float64\n",
            " 15  compactness error        569 non-null    float64\n",
            " 16  concavity error          569 non-null    float64\n",
            " 17  concave points error     569 non-null    float64\n",
            " 18  symmetry error           569 non-null    float64\n",
            " 19  fractal dimension error  569 non-null    float64\n",
            " 20  worst radius             569 non-null    float64\n",
            " 21  worst texture            569 non-null    float64\n",
            " 22  worst perimeter          569 non-null    float64\n",
            " 23  worst area               569 non-null    float64\n",
            " 24  worst smoothness         569 non-null    float64\n",
            " 25  worst compactness        569 non-null    float64\n",
            " 26  worst concavity          569 non-null    float64\n",
            " 27  worst concave points     569 non-null    float64\n",
            " 28  worst symmetry           569 non-null    float64\n",
            " 29  worst fractal dimension  569 non-null    float64\n",
            " 30  label                    569 non-null    int64  \n",
            "dtypes: float64(30), int64(1)\n",
            "memory usage: 137.9 KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# checking for missing values\n",
        "data_frame.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "zaB3sRIuG_G9",
        "outputId": "0a9d12ed-ced3-4302-9406-f5a0871fc60f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "mean radius                0\n",
              "mean texture               0\n",
              "mean perimeter             0\n",
              "mean area                  0\n",
              "mean smoothness            0\n",
              "mean compactness           0\n",
              "mean concavity             0\n",
              "mean concave points        0\n",
              "mean symmetry              0\n",
              "mean fractal dimension     0\n",
              "radius error               0\n",
              "texture error              0\n",
              "perimeter error            0\n",
              "area error                 0\n",
              "smoothness error           0\n",
              "compactness error          0\n",
              "concavity error            0\n",
              "concave points error       0\n",
              "symmetry error             0\n",
              "fractal dimension error    0\n",
              "worst radius               0\n",
              "worst texture              0\n",
              "worst perimeter            0\n",
              "worst area                 0\n",
              "worst smoothness           0\n",
              "worst compactness          0\n",
              "worst concavity            0\n",
              "worst concave points       0\n",
              "worst symmetry             0\n",
              "worst fractal dimension    0\n",
              "label                      0\n",
              "dtype: int64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>mean radius</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean texture</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean perimeter</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean area</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean smoothness</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean compactness</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean concavity</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean concave points</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean symmetry</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean fractal dimension</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>radius error</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>texture error</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>perimeter error</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>area error</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>smoothness error</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>compactness error</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>concavity error</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>concave points error</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>symmetry error</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>fractal dimension error</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>worst radius</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>worst texture</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>worst perimeter</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>worst area</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>worst smoothness</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>worst compactness</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>worst concavity</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>worst concave points</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>worst symmetry</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>worst fractal dimension</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>label</th>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 63
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# checking the distribution of Target Varibale\n",
        "data_frame['label'].value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 178
        },
        "id": "ulfusZUtHi6E",
        "outputId": "c1dc570a-7463-48c1-aa77-c2c0a862180f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "label\n",
              "1    357\n",
              "0    212\n",
              "Name: count, dtype: int64"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>count</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>label</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>357</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>212</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div><br><label><b>dtype:</b> int64</label>"
            ]
          },
          "metadata": {},
          "execution_count": 64
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data_frame.groupby('label').mean()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 227
        },
        "id": "mY6R3MaaH0Xy",
        "outputId": "08cb94f3-57ab-4e20-c9b2-0c1a391ac283"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "       mean radius  mean texture  mean perimeter   mean area  mean smoothness  \\\n",
              "label                                                                           \n",
              "0        17.462830     21.604906      115.365377  978.376415         0.102898   \n",
              "1        12.146524     17.914762       78.075406  462.790196         0.092478   \n",
              "\n",
              "       mean compactness  mean concavity  mean concave points  mean symmetry  \\\n",
              "label                                                                         \n",
              "0              0.145188        0.160775             0.087990       0.192909   \n",
              "1              0.080085        0.046058             0.025717       0.174186   \n",
              "\n",
              "       mean fractal dimension  ...  worst radius  worst texture  \\\n",
              "label                          ...                                \n",
              "0                    0.062680  ...     21.134811      29.318208   \n",
              "1                    0.062867  ...     13.379801      23.515070   \n",
              "\n",
              "       worst perimeter   worst area  worst smoothness  worst compactness  \\\n",
              "label                                                                      \n",
              "0           141.370330  1422.286321          0.144845           0.374824   \n",
              "1            87.005938   558.899440          0.124959           0.182673   \n",
              "\n",
              "       worst concavity  worst concave points  worst symmetry  \\\n",
              "label                                                          \n",
              "0             0.450606              0.182237        0.323468   \n",
              "1             0.166238              0.074444        0.270246   \n",
              "\n",
              "       worst fractal dimension  \n",
              "label                           \n",
              "0                     0.091530  \n",
              "1                     0.079442  \n",
              "\n",
              "[2 rows x 30 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-b0985104-2809-4d0b-b270-49d0f30f51e5\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>mean radius</th>\n",
              "      <th>mean texture</th>\n",
              "      <th>mean perimeter</th>\n",
              "      <th>mean area</th>\n",
              "      <th>mean smoothness</th>\n",
              "      <th>mean compactness</th>\n",
              "      <th>mean concavity</th>\n",
              "      <th>mean concave points</th>\n",
              "      <th>mean symmetry</th>\n",
              "      <th>mean fractal dimension</th>\n",
              "      <th>...</th>\n",
              "      <th>worst radius</th>\n",
              "      <th>worst texture</th>\n",
              "      <th>worst perimeter</th>\n",
              "      <th>worst area</th>\n",
              "      <th>worst smoothness</th>\n",
              "      <th>worst compactness</th>\n",
              "      <th>worst concavity</th>\n",
              "      <th>worst concave points</th>\n",
              "      <th>worst symmetry</th>\n",
              "      <th>worst fractal dimension</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>label</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>17.462830</td>\n",
              "      <td>21.604906</td>\n",
              "      <td>115.365377</td>\n",
              "      <td>978.376415</td>\n",
              "      <td>0.102898</td>\n",
              "      <td>0.145188</td>\n",
              "      <td>0.160775</td>\n",
              "      <td>0.087990</td>\n",
              "      <td>0.192909</td>\n",
              "      <td>0.062680</td>\n",
              "      <td>...</td>\n",
              "      <td>21.134811</td>\n",
              "      <td>29.318208</td>\n",
              "      <td>141.370330</td>\n",
              "      <td>1422.286321</td>\n",
              "      <td>0.144845</td>\n",
              "      <td>0.374824</td>\n",
              "      <td>0.450606</td>\n",
              "      <td>0.182237</td>\n",
              "      <td>0.323468</td>\n",
              "      <td>0.091530</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>12.146524</td>\n",
              "      <td>17.914762</td>\n",
              "      <td>78.075406</td>\n",
              "      <td>462.790196</td>\n",
              "      <td>0.092478</td>\n",
              "      <td>0.080085</td>\n",
              "      <td>0.046058</td>\n",
              "      <td>0.025717</td>\n",
              "      <td>0.174186</td>\n",
              "      <td>0.062867</td>\n",
              "      <td>...</td>\n",
              "      <td>13.379801</td>\n",
              "      <td>23.515070</td>\n",
              "      <td>87.005938</td>\n",
              "      <td>558.899440</td>\n",
              "      <td>0.124959</td>\n",
              "      <td>0.182673</td>\n",
              "      <td>0.166238</td>\n",
              "      <td>0.074444</td>\n",
              "      <td>0.270246</td>\n",
              "      <td>0.079442</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>2 rows × 30 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-b0985104-2809-4d0b-b270-49d0f30f51e5')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-b0985104-2809-4d0b-b270-49d0f30f51e5 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-b0985104-2809-4d0b-b270-49d0f30f51e5');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe"
            }
          },
          "metadata": {},
          "execution_count": 65
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "l3x-EEJPImSb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "f6cf5c31"
      },
      "source": [
        "X = data_frame.drop(columns='label', axis=1)\n",
        "Y = data_frame['label']"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(X)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8AfC7J9hI0am",
        "outputId": "b06a266b-ee7c-474b-91f6-14b09e085d95"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "     mean radius  mean texture  mean perimeter  mean area  mean smoothness  \\\n",
            "0          17.99         10.38          122.80     1001.0          0.11840   \n",
            "1          20.57         17.77          132.90     1326.0          0.08474   \n",
            "2          19.69         21.25          130.00     1203.0          0.10960   \n",
            "3          11.42         20.38           77.58      386.1          0.14250   \n",
            "4          20.29         14.34          135.10     1297.0          0.10030   \n",
            "..           ...           ...             ...        ...              ...   \n",
            "564        21.56         22.39          142.00     1479.0          0.11100   \n",
            "565        20.13         28.25          131.20     1261.0          0.09780   \n",
            "566        16.60         28.08          108.30      858.1          0.08455   \n",
            "567        20.60         29.33          140.10     1265.0          0.11780   \n",
            "568         7.76         24.54           47.92      181.0          0.05263   \n",
            "\n",
            "     mean compactness  mean concavity  mean concave points  mean symmetry  \\\n",
            "0             0.27760         0.30010              0.14710         0.2419   \n",
            "1             0.07864         0.08690              0.07017         0.1812   \n",
            "2             0.15990         0.19740              0.12790         0.2069   \n",
            "3             0.28390         0.24140              0.10520         0.2597   \n",
            "4             0.13280         0.19800              0.10430         0.1809   \n",
            "..                ...             ...                  ...            ...   \n",
            "564           0.11590         0.24390              0.13890         0.1726   \n",
            "565           0.10340         0.14400              0.09791         0.1752   \n",
            "566           0.10230         0.09251              0.05302         0.1590   \n",
            "567           0.27700         0.35140              0.15200         0.2397   \n",
            "568           0.04362         0.00000              0.00000         0.1587   \n",
            "\n",
            "     mean fractal dimension  ...  worst radius  worst texture  \\\n",
            "0                   0.07871  ...        25.380          17.33   \n",
            "1                   0.05667  ...        24.990          23.41   \n",
            "2                   0.05999  ...        23.570          25.53   \n",
            "3                   0.09744  ...        14.910          26.50   \n",
            "4                   0.05883  ...        22.540          16.67   \n",
            "..                      ...  ...           ...            ...   \n",
            "564                 0.05623  ...        25.450          26.40   \n",
            "565                 0.05533  ...        23.690          38.25   \n",
            "566                 0.05648  ...        18.980          34.12   \n",
            "567                 0.07016  ...        25.740          39.42   \n",
            "568                 0.05884  ...         9.456          30.37   \n",
            "\n",
            "     worst perimeter  worst area  worst smoothness  worst compactness  \\\n",
            "0             184.60      2019.0           0.16220            0.66560   \n",
            "1             158.80      1956.0           0.12380            0.18660   \n",
            "2             152.50      1709.0           0.14440            0.42450   \n",
            "3              98.87       567.7           0.20980            0.86630   \n",
            "4             152.20      1575.0           0.13740            0.20500   \n",
            "..               ...         ...               ...                ...   \n",
            "564           166.10      2027.0           0.14100            0.21130   \n",
            "565           155.00      1731.0           0.11660            0.19220   \n",
            "566           126.70      1124.0           0.11390            0.30940   \n",
            "567           184.60      1821.0           0.16500            0.86810   \n",
            "568            59.16       268.6           0.08996            0.06444   \n",
            "\n",
            "     worst concavity  worst concave points  worst symmetry  \\\n",
            "0             0.7119                0.2654          0.4601   \n",
            "1             0.2416                0.1860          0.2750   \n",
            "2             0.4504                0.2430          0.3613   \n",
            "3             0.6869                0.2575          0.6638   \n",
            "4             0.4000                0.1625          0.2364   \n",
            "..               ...                   ...             ...   \n",
            "564           0.4107                0.2216          0.2060   \n",
            "565           0.3215                0.1628          0.2572   \n",
            "566           0.3403                0.1418          0.2218   \n",
            "567           0.9387                0.2650          0.4087   \n",
            "568           0.0000                0.0000          0.2871   \n",
            "\n",
            "     worst fractal dimension  \n",
            "0                    0.11890  \n",
            "1                    0.08902  \n",
            "2                    0.08758  \n",
            "3                    0.17300  \n",
            "4                    0.07678  \n",
            "..                       ...  \n",
            "564                  0.07115  \n",
            "565                  0.06637  \n",
            "566                  0.07820  \n",
            "567                  0.12400  \n",
            "568                  0.07039  \n",
            "\n",
            "[569 rows x 30 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(Y)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CLgc1b2rI73Z",
        "outputId": "ae966dca-3f5f-46e1-b025-a0a7e5eed83f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0      0\n",
            "1      0\n",
            "2      0\n",
            "3      0\n",
            "4      0\n",
            "      ..\n",
            "564    0\n",
            "565    0\n",
            "566    0\n",
            "567    0\n",
            "568    1\n",
            "Name: label, Length: 569, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)"
      ],
      "metadata": {
        "id": "-9w8BMCxJD37"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(X.shape, X_train.shape, X_test.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6HqezrSHJNN-",
        "outputId": "436880ae-b8bc-41ad-9945-987f69846f39"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(569, 30) (455, 30) (114, 30)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.preprocessing import StandardScaler"
      ],
      "metadata": {
        "id": "sbwVp_BtJjm0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "scaler = StandardScaler()\n",
        "\n",
        "X_train_std = scaler.fit_transform(X_train)\n",
        "\n",
        "X_test_std = scaler.transform(X_test)"
      ],
      "metadata": {
        "id": "KTUx-vJpJsZP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# importing tensorflow and Keras\n",
        "import tensorflow as tf\n",
        "tf.random.set_seed(3)\n",
        "from tensorflow import keras"
      ],
      "metadata": {
        "id": "5omh2-YAJu7u"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# setting up the layers of Neural Network\n",
        "\n",
        "model = keras.Sequential([\n",
        "                          keras.layers.Flatten(input_shape=(30,)),\n",
        "                          keras.layers.Dense(20, activation='relu'),\n",
        "                          keras.layers.Dense(2, activation='sigmoid')\n",
        "])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BYuV0ptrKL5f",
        "outputId": "b719bf29-3644-44a0-f35b-d68bf0bb88ff"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/keras/src/layers/reshaping/flatten.py:37: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
            "  super().__init__(**kwargs)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# compiling the Neural Network\n",
        "\n",
        "model.compile(optimizer='adam',\n",
        "              loss='sparse_categorical_crossentropy',\n",
        "              metrics=['accuracy'])"
      ],
      "metadata": {
        "id": "cNj78nKDKOfK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# training the Meural Network\n",
        "\n",
        "history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VlfIVlgRKUHL",
        "outputId": "a8446862-5c65-4ad5-e905-1a61459e40f6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "\u001b[1m13/13\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 15ms/step - accuracy: 0.7250 - loss: 0.5619 - val_accuracy: 0.7826 - val_loss: 0.4362\n",
            "Epoch 2/10\n",
            "\u001b[1m13/13\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.8358 - loss: 0.4132 - val_accuracy: 0.8478 - val_loss: 0.3318\n",
            "Epoch 3/10\n",
            "\u001b[1m13/13\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.8657 - loss: 0.3207 - val_accuracy: 0.9348 - val_loss: 0.2697\n",
            "Epoch 4/10\n",
            "\u001b[1m13/13\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9014 - loss: 0.2609 - val_accuracy: 0.9348 - val_loss: 0.2308\n",
            "Epoch 5/10\n",
            "\u001b[1m13/13\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9069 - loss: 0.2201 - val_accuracy: 0.9565 - val_loss: 0.2050\n",
            "Epoch 6/10\n",
            "\u001b[1m13/13\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9332 - loss: 0.1904 - val_accuracy: 0.9565 - val_loss: 0.1870\n",
            "Epoch 7/10\n",
            "\u001b[1m13/13\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9625 - loss: 0.1681 - val_accuracy: 0.9348 - val_loss: 0.1737\n",
            "Epoch 8/10\n",
            "\u001b[1m13/13\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9621 - loss: 0.1507 - val_accuracy: 0.9348 - val_loss: 0.1632\n",
            "Epoch 9/10\n",
            "\u001b[1m13/13\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9642 - loss: 0.1368 - val_accuracy: 0.9348 - val_loss: 0.1544\n",
            "Epoch 10/10\n",
            "\u001b[1m13/13\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9706 - loss: 0.1254 - val_accuracy: 0.9348 - val_loss: 0.1472\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.plot(history.history['accuracy'])\n",
        "plt.plot(history.history['val_accuracy'])\n",
        "\n",
        "plt.title('model accuracy')\n",
        "plt.ylabel('accuracy')\n",
        "plt.xlabel('epoch')\n",
        "\n",
        "plt.legend(['training data', 'validation data'], loc = 'lower right')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 489
        },
        "id": "n1kwU7HqKb0J",
        "outputId": "66ae5e20-144b-4ff0-9916-a15bbc6e24d6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.legend.Legend at 0x798b503bd640>"
            ]
          },
          "metadata": {},
          "execution_count": 77
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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FhZm/Otf/zJiYmIcx5w6dcq4/fbbDR8fH6NSpUrGv//9b2Pbtm15ToPP6+dsGIaxY8cOo0ePHoafn58RHBxsjBw50ti8eXOeP69t27YZ/fv3N4KCggwvLy+jQYMGxqRJk3JdMz093ahUqZIRGBhonD9//oo/NxFnYjGMEhzdKCIipsrKyqJ69er07duXDz/80OxyRMoMjQESEanAvv/+e06dOpVjYLWIgFqAREQqoDVr1rBlyxamTp1KcHBwjhtAiohagEREKqR3332XBx98kKpVq/Lpp5+aXY5ImaMWIBEREXE6agESERERp6MAJCIiIk5HN0LMg81m49ixY/j7+xdptWcREREpPYZhcO7cOapXr55rvb1/UgDKw7Fjx4iIiDC7DBERESmEI0eOUKNGjSseowCUh+xb+R85coSAgACTqxEREZH8SEpKIiIiwvE9fiUKQHnI7vYKCAhQABIRESln8jN8RYOgRURExOkoAImIiIjTUQASERERp6MAJCIiIk5HAUhEREScjgKQiIiIOB0FIBEREXE6CkAiIiLidBSARERExOkoAImIiIjTUQASERERp6MAJCIiIk5HAUhERERKTZbVxqGEFE6dSze1Dq0GLyIiIsXKZjM4npTGwfgUDsSnOP48kJDCkdOpZFoNnujZgIevrWtajQpAIiIiUmCGYXAqOZ0Dp1I4mJDCgfhUDsQnczA+lYMJKaRn2S57rqebC6kZWaVYbW4KQCIiInJZZ1IyOJCQcknQudiqk5Jhvex5bi4WalbxIaqKL5HBvkRdeEQG+1ItwAsXF0spfoo86jP13UVERMR059IyORifyv5LWnCyg07i+czLnudigfBK3kQF+xFVxSdH0AkP8sbNtewONVYAEhERcQLnM6wcTLC33Oy/0IKTHXTikzOueG61QC8iq/gSFeKbo0UnorI3nm6upfQJipcCkIiISAWRnmXlyOlUDsSn5go6xxPTrnhusJ8nUcE+uYJOZBVfvD3KZ8i5EgUgERGRciTLauPomfMcuNCa4wg6CSnEnjmPzbj8uYHe7hfH4uQIOj74e7mX3ocoAxSAREREypjsaeQHTqU4gk72wOPDp1PJukLK8fVwzTno+JKgU8nXoxQ/RdmmACQiImICwzA4dS7dMdj40qBzKCH1qtPIIy+03EQF++Xougrx88RiMXeGVXmgACQiIlJC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V6FagJo2bcrmzZuJiooiOjqaV155BQ8PD95//31q165d3DVKaTuwAg6tBFcP6DbWsXv7sST2n0rB082FG5uEmligiIhI0RQqAE2cOJGUlBQApkyZQp8+fejatStVqlRh7ty5xVqglDLDgF+ft2+3vQcCazheyh783KNRKP5eut2BiIiUX4UKQD179nRs161bl127dnH69GkqVaqkxS/Lu72/wNF14OYNXcY4dlttBgs227s6b2mp7i8RESnfim3hrsqVNR263Lu09af9SPC/2M219sBp4pLSCfByo3sD3fRQRETKt0INgpYKauePcGILePhB58dzvLTgwuDn3s2q4enmakJxIiIixUcBSOxs1ot3fe7wEPhWcbyUnmXl560nAHV/iYhIxaAAJHbb58OpneAVCB0fzvHS73viSTyfSWiAJ9FRVS5zARERkfJDAUjAmnWx9afTI+AdlOPl7NlffZtXx9VFg9xFRKT8UwAS2PIVnI4BnyoQ/UCOl5LTs1i6Mw6Afi3DzahORESk2CkAObusDFjxsn278+Pg6Z/j5V+2nyAt00btYF+ahgeUfn0iIiIlQAHI2W38DM4eBr9QaHdfrpd/2HTx3j+6x5OIiFQUCkDOLDMNfn/Nvt31P+Dhk+Pl+OR0Vu6LB+AWrf0lIiIViAKQM1v/EZw7BgE1oM3wXC//vPU4VptB8xqB1A7xK/36RERESogCkLPKSIE/XrdvX/MEuHnmOsTR/aXWHxERqWAUgJzV2g8g5RRUioSWQ3K9fOR0KusPncFigb4KQCIiUsEoADmjtCRYNcO+fc04cM29snv2wqcda1chNMCrFIsTEREpeQpAzuivd+H8GQiuD80H5HnIggvdX/209IWIiFRACkDOJvU0rJ5p3+4+DlxyL2y660QSu+PO4eHqQq8m1Uq5QBERkZKnAORsVs+E9CSo2gQa98/zkOzBz90bhBDok7t7TEREpLxTAHImKfHw1yz79nUTwCX3r99mMy7p/tLSFyIiUjEpADmTlf8HmSlQvRU06J3nIRsOnyH27Hl8PVy5vlHVUi5QRESkdCgAOYuk47Duv/btayfCZZa1yO7+6tk0DC/33OODREREKgIFIGfxx+uQlQYRHaDu9Xkekmm1sXDrcUDdXyIiUrEpADmDs4dh/cf27esmXLb1Z+W+eE6nZBDs50HnOlVKrz4REZFSpgDkDH5/FWyZENXN/riM7MHPfZpXx81VfzVERKTi0rdcRZcQAxvn2LevnXjZw85nWFm8/QQAt+jmhyIiUsEpAFV0K14Bwwp1b4Ca0Zc9bOnOOFIzrERU9qZVRFDp1SciImICBaCK7OQu2DLXvn3dhCsemj37q1+LcCyXGSMkIiJSUSgAVWTLpwEGNOxjv/fPZZxNzWDFnpOA1v4SERHnoABUUZ3YCju+Byxw7dNXPPR/206QaTVoVC2AeqH+pVKeiIiImRSAKqrfXrT/2fQ2CG1yxUN/2BQLqPVHRESchwJQRXR0Pez+GSwu0H38FQ89nnieNQdOA9C3hQKQiIg4BwWgiui3F+x/Nh8EwfWueOiPm49hGNA+sjLhQd6lUJyIiIj5FIAqmkN/QswycHGDa5686uHZs7907x8REXEmCkAViWHArxdaf1rdDZWjrnj4vpPn2H4sCTcXC72bVSuFAkVERMoGBaCK5MAKOLQSXD2g29irHp699EW3+iFU9vUo6epERETKDAWgisIw4Nfn7dtt74HAGlc53OCHzRdufqjuLxERcTIKQBXF3l/g6Dpw84YuY656+OajiRxKSMXb3ZUejUJLoUAREZGyQwGoIri09af9SPC/eqDJvvfPDY1D8fV0K8nqREREyhwFoIpg549wYgt4+EHnx696uNVm8OPm44C6v0RExDkpAJV3NuvFuz53eBB8q1z1lNUxCcQnpxPk407XeiElXKCIiEjZowBU3m2fD6d2glcgdByVr1Oyu796N6uGh5v+CoiIiPPRt195Zs262PrT6RHwDrrqKWmZVhZtOwFAPy19ISIiTkoBqDzbMhdOx4B3ZYh+IF+nLN99knPpWVQL9KJdZOUSLlBERKRsUgAqr7IyYMVL9u0uo8HTP1+nOZa+aFEdFxdLSVUnIiJSpikAlVcbP4Ozh8EvFNrdl69TktIyWbbrJAD9WoaXZHUiIiJlmgJQeZSZBr+/Zt/u+h/w8MnXaYu3nSAjy0a9qn40qpa/FiMREZGKSAGoPFr/EZw7BgE1oM3wfJ+24JKlLywWdX+JiIjzUgAqbzJS4I/X7dvXPAFunvk67eS5NFbtiwfglhbq/hIREeemAFTerP0AUk5BpUhoOSTfpy3cchybAa1qBlGzSv66zERERCoqBaDyJC0JVs2wb18zDlzd831q9uwv3ftHREREAah8+etdOH8GgutD8wH5Pu1QQgqbjpzFxQI3N1cAEhERUQAqL1JPw+qZ9u3u48DFNd+nLrjQ+tO5bjAh/vkbMyQiIlKRmR6A3n77bSIjI/Hy8iI6Opq1a9de8fgZM2bQoEEDvL29iYiIYPTo0aSlpRXpmuXC6pmQngRVm0Dj/vk+zTAMvr+w9pfu/SMiImJnagCaO3cuY8aMYfLkyWzYsIEWLVrQs2dPTp48mefxX3zxBePGjWPy5Mns3LmTDz/8kLlz5/L0008X+prlQko8/DXLvn3dBHDJ/69t+7EkYk6l4OHmQs8moSVUoIiISPliagCaPn06I0eOZMSIETRu3JhZs2bh4+PD7Nmz8zz+zz//pHPnztx5551ERkZy4403Mnjw4BwtPAW9Zrmw8v8gMwWqt4IGvQt0ava9f3o0qoq/V/4HTYuIiFRkpgWgjIwM1q9fT48ePS4W4+JCjx49WL16dZ7ndOrUifXr1zsCz/79+/n555/p3bt3oa8JkJ6eTlJSUo5HmZF0HNb917597UQowA0MbTbDMf5H9/4RERG5yM2sN46Pj8dqtRIamrNbJjQ0lF27duV5zp133kl8fDxdunTBMAyysrJ44IEHHF1ghbkmwLRp03juueeK+IlKyB+vQ1YaRHSAutcX6NS1B09zIikNfy83ujcIKaECRUREyh/TB0EXxPLly3nxxRd555132LBhA/PmzWPhwoVMnTq1SNcdP348iYmJjseRI0eKqeIiOnsY1n9s375uQoFaf+DivX9uahqGl3v+Z42JiIhUdKa1AAUHB+Pq6kpcXFyO/XFxcYSFheV5zqRJk7j77ru57z776ufNmjUjJSWF+++/nwkTJhTqmgCenp54epbB6eG/vwq2TIjqZn8UQEaWjZ+3Hgc0+0tEROSfTGsB8vDwoE2bNixbtsyxz2azsWzZMjp27JjnOampqbj8YwaUq6u9ZcMwjEJds8xKiIGNc+zb104s8Om/7zlF4vlMQvw96VC7SjEXJyIiUr6Z1gIEMGbMGIYNG0bbtm1p3749M2bMICUlhREjRgAwdOhQwsPDmTZtGgB9+/Zl+vTptGrViujoaPbt28ekSZPo27evIwhd7ZrlxopXwLBC3RugZnSBT//hwuyvvs2r4+qild9FREQuZWoAGjhwIKdOneKZZ57hxIkTtGzZkkWLFjkGMR8+fDhHi8/EiROxWCxMnDiR2NhYQkJC6Nu3Ly+88EK+r1kunNoNW+bat6+bUODTU9KzWLLjBAD9WmrpCxERkX+yGIZhmF1EWZOUlERgYCCJiYkEBASUfgHfDIft86FhHxg0p8Cnf78xlsfnbiKyig+/je2OpYCDp0VERMqjgnx/l6tZYE7hxFZ7+MEC1z591cPz8sOFpS9uaRmu8CMiIpIHBaCy5rcX7X82vQ1CmxT49ITkdH7fGw+o+0tERORyFIDKktj1sPtnsLhA9/GFusTP205gtRk0Cw+kTohfMRcoIiJSMSgAlSW/XhjM3XwQBNcr1CUWOFZ+V+uPiIjI5SgAlRWHVkPMMnBxg2ueLNQljp5JZd3BM1gs0Ke5ApCIiMjlKACVBYYBvz5v3251F1SOKtRlftxsv/Nzh6gqhAV6FVd1IiIiFY4CUFlwYAUcWgmuHtDtiUJf5gd1f4mIiOSLApDZLm39aXsPBNYo1GV2nzjHrhPncHe1cFPTasVYoIiISMWjAGS2vUvg6Dpw84YuYwp9mQWb7a0/3RtUJdDHvbiqExERqZAUgMxkGPDrVPt2+5HgX7jlOgzD4IdN9rW/1P0lIiJydQpAZtr5I5zYAh5+0PnxQl9mw+GzHD1zHl8PV65vWI7WPBMRETGJApBZbNaLd33u8CD4Vin0pbLv/dOzSRjeHq7FUZ2IiEiFpgBklu3z4dRO8AqEjqMKfZksq42fttinv9+i7i8REZF8UQAygzXrYutPp0fAO6jQl1q5L56ElAyq+HrQuW5w8dQnIiJSwSkAmWHLXDgdA96VIfqBIl1qwYXBzzc3r4a7q36dIiIi+aFvzNKWlQErXrJvdxkNnv6FvtT5DCuLt58ANPtLRESkIBSAStvGz+DsYfALhXb3FelSy3bFkZJhpUYlb1rXrFRMBYqIiFR8CkClKTMNfn/Nvt31P+DhU6TLZd/755YW1bFYLEWtTkRExGkoAJWm9R/BuWMQUAPaDC/SpRJTM1m++yQA/VqGF0NxIiIizkMBqDR5+IJPMFzzBLh5FulS/9t2nEyrQcMwfxqEFX4ckYiIiDNyM7sAp9J6KDS5rcjhBy7p/tLgZxERkQJTACptnn5FvsSJxDT+OpAAQN/mCkAiIiIFpS6wcuinLccwDGhbqxIRlYs2kFpERMQZKQCVQ46V31tp8LOIiEhhKACVMzGnktkam4ibi4Wbm1UzuxwREZFySQGonMle+qJrvWAq+3qYXI2IiEj5pABUjhiGwYLNF7q/dO8fERGRQlMAKke2xiZyID4FL3cXbmgcanY5IiIi5ZYCUDmSPfj5hsZh+HrqDgYiIiKFpQBUTlhtBj9md3+10L1/REREikIBqJxYsz+Bk+fSCfR2p1v9ELPLERERKdcUgMqJ7O6v3s2q4eGmX5uIiEhR6Ju0HEjPsvLztuMA9NPaXyIiIkWmAFQOLN99inNpWYQFeNE+srLZ5YiIiJR7CkDlwIJLVn53cbGYXI2IiEj5pwBUxp1Ly2TpzjgAbtHsLxERkWKhAFTGLd4eR3qWjTohvjSpHmB2OSIiIhWC7qZXxv2wKRawL31hsaj7S0TKPqvVSmZmptllSAXk7u6Oq6trsVxLAagMO3UunVX74gF1f4lI2WcYBidOnODs2bNmlyIVWFBQEGFhYUVuFFAAKsMWbjmGzYAWEUFEBvuaXY6IyBVlh5+qVavi4+OjVmspVoZhkJqaysmTJwGoVq1aka6nAFSG/aClL0SknLBarY7wU6VKFbPLkQrK29sbgJMnT1K1atUidYdpEHQZdTghlY2Hz+JigT7Ni5ZyRURKWvaYHx8fH5MrkYou++9YUceZKQCVUQs22wc/d6oTTNUAL5OrERHJH3V7SUkrrr9jCkBlkGEYfH/JzQ9FRKT8iIyMZMaMGfk+fvny5VgsFlMGj3/88ccEBQWV+vuWBQpAZdDO4+fYdzIZDzcXejUNM7scEZEKrXv37jz++OPFdr1169Zx//335/v4Tp06cfz4cQIDA4uthpJU0IBXVmkQdBn0w4Xur+sbViXAy93kakRExDAMrFYrbm5X/9oMCQkp0LU9PDwIC9P/7JY2tQCVMTabwY8Xur+08ruISMkaPnw4K1as4I033sBisWCxWDh48KCjW+p///sfbdq0wdPTk5UrVxITE0O/fv0IDQ3Fz8+Pdu3asXTp0hzX/GcLicVi4b///S/9+/fHx8eHevXqsWDBAsfr/+wCy+6WWrx4MY0aNcLPz49evXpx/PhxxzlZWVk8+uijBAUFUaVKFZ566imGDRvGrbfeesXP+/HHH1OzZk18fHzo378/CQkJOV6/2ufr3r07hw4dYvTo0Y6fF0BCQgKDBw8mPDwcHx8fmjVrxpdfflmQX0WpUwAqY/4+dIZjiWn4e7rRvUFVs8sRESk0wzBIzcgy5WEYRr5qfOONN+jYsSMjR47k+PHjHD9+nIiICMfr48aN46WXXmLnzp00b96c5ORkevfuzbJly9i4cSO9evWib9++HD58+Irv89xzzzFgwAC2bNlC7969GTJkCKdPn77s8ampqbz22mt89tln/P777xw+fJixY8c6Xn/55ZeZM2cOH330EatWrSIpKYnvv//+ijWsWbOGe++9l1GjRrFp0yauvfZann/++RzHXO3zzZs3jxo1ajBlyhTHzwsgLS2NNm3asHDhQrZt28b999/P3Xffzdq1a69Yk5nUBVbGZC990atpGF7uxXO7bxERM5zPtNL4mcWmvPeOKT3x8bj6V1xgYCAeHh74+Pjk2Q01ZcoUbrjhBsfzypUr06JFC8fzqVOnMn/+fBYsWMCoUaMu+z7Dhw9n8ODBALz44ou8+eabrF27ll69euV5fGZmJrNmzaJOnToAjBo1iilTpjhef+uttxg/fjz9+/cHYObMmfz8889X/KxvvPEGvXr14sknnwSgfv36/PnnnyxatMhxTIsWLa74+SpXroyrqyv+/v45fl7h4eE5AtojjzzC4sWL+frrr2nfvv0V6zKLWoDKkIwsGwu32tN0v5bhJlcjIiJt27bN8Tw5OZmxY8fSqFEjgoKC8PPzY+fOnVdtAWrevLlj29fXl4CAAMcdjfPi4+PjCD9gv+tx9vGJiYnExcXlCBaurq60adPmijXs3LmT6OjoHPs6duxYLJ/ParUydepUmjVrRuXKlfHz82Px4sVXPc9MagEqQ1buO8XZ1EyC/TzpWEd3UhWR8s3b3ZUdU3qa9t7Fwdc35zJEY8eOZcmSJbz22mvUrVsXb29v/vWvf5GRkXHF67i755zQYrFYsNlsBTo+v916RVHYz/fqq6/yxhtvMGPGDJo1a4avry+PP/74Vc8zkwJQGfLDhcHPfVtUw9VFNxMTkfLNYrHkqxvKbB4eHlit1nwdu2rVKoYPH+7oekpOTubgwYMlWF1ugYGBhIaGsm7dOrp16wbYW2A2bNhAy5YtL3teo0aNWLNmTY59f/31V47n+fl8ef28Vq1aRb9+/bjrrrsAsNls7Nmzh8aNGxfmI5YKdYGVEakZWfyyPQ5Q95eISGmKjIxkzZo1HDx4kPj4+Cu2zNSrV4958+axadMmNm/ezJ133nnF40vKI488wrRp0/jhhx/YvXs3jz32GGfOnLniXZIfffRRFi1axGuvvcbevXuZOXNmjvE/kL/PFxkZye+//05sbCzx8fGO85YsWcKff/7Jzp07+fe//01cXFzxf/BipABURizZEcf5TCu1qvjQokb5uBmWiEhFMHbsWFxdXWncuDEhISFXHLcyffp0KlWqRKdOnejbty89e/akdevWpVit3VNPPcXgwYMZOnQoHTt2xM/Pj549e+Lldfmlkzp06MAHH3zAG2+8QYsWLfjll1+YOHFijmPy8/mmTJnCwYMHqVOnjuOeRxMnTqR169b07NmT7t27ExYWdtUp+WazGKXRqVjOJCUlERgYSGJiIgEBAaXynvd+vI5lu07y6HV1GXNjg1J5TxGR4pKWlsaBAweIioq64pewlAybzUajRo0YMGAAU6dONbucEnWlv2sF+f4u+52zTuBMSgYr9pwCtPaXiIhc3aFDh/jll1+45pprSE9PZ+bMmRw4cIA777zT7NLKDXWBlQE/bztOls2gSfUA6lb1N7scEREp41xcXPj4449p164dnTt3ZuvWrSxdupRGjRqZXVq5oRagMuAHLX0hIiIFEBERwapVq8wuo1xTC5DJYs+eZ+2B01gs0LeFApCIiEhpUAAy2Y+b7a0/7SMrUy3Q2+RqREREnIMCkMkudn/p3j8iIiKlRQHIRHvizrHzeBLurhZuapp7ET4REREpGQpAJlpwofXnmvohVPL1MLkaERER51EmAtDbb79NZGQkXl5eREdHs3bt2sse2717dywWS67HzTff7Dhm+PDhuV7v1atXaXyUfDMMgx82xwJwi7q/RERESpXpAWju3LmMGTOGyZMns2HDBlq0aEHPnj05efJknsfPmzeP48ePOx7btm3D1dWVO+64I8dxvXr1ynHcl19+WRofJ982HjnLkdPn8fFwpUejqmaXIyIiRRAZGcmMGTMczy0WC99///1ljz948CAWi4VNmzYV6X2L6zqFMXz48DK/3MWVmB6Apk+fzsiRIxkxYgSNGzdm1qxZ+Pj4MHv27DyPr1y5MmFhYY7HkiVL8PHxyRWAPD09cxxXqVKl0vg4+Zbd/dWzSVi5WC1ZRETy7/jx49x0003Fes28AkdERATHjx+nadOmxfpeJcHMsJYXUwNQRkYG69evp0ePHo59Li4u9OjRg9WrV+frGh9++CGDBg3C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Vmzbto1Zs2YpAJng66+/Zs6cOXzxxRc0adKETZs28fjjj1O9enX9PopAAagUBAcH4+rqSlxcXI79cXFxhIWFmVSVjBo1ip9++onff/+dGjVqmF2O01q/fj0nT56kdevWjn1Wq5Xff/+dmTNnkp6ejqurq4kVOp9q1arRuHHjHPsaNWrEd999Z1JFzu2JJ55g3LhxDBo0CIBmzZpx6NAhpk2bpgBUBBoDVAo8PDxo06YNy5Ytc+yz2WwsW7aMjh07mliZczIMg1GjRjF//nx+/fVXoqKizC7JqV1//fVs3bqVTZs2OR5t27ZlyJAhbNq0SeHHBJ07d851a4g9e/ZQq1Ytkypybqmpqbi45Py6dnV1xWazmVRRxaAWoFIyZswYhg0bRtu2bWnfvj0zZswgJSWFESNGmF2a03n44Yf54osv+OGHH/D39+fEiRMABAYG4u3tbXJ1zsff3z/X+CtfX1+qVKmicVkmGT16NJ06deLFF19kwIABrF27lvfff5/333/f7NKcUt++fXnhhReoWbMmTZo0YePGjUyfPp177rnH7NLKNU2DL0UzZ87k1Vdf5cSJE7Rs2ZI333yT6Ohos8tyOhaLJc/9H330EcOHDy/dYiRP3bt31zR4k/3000+MHz+evXv3EhUVxZgxYxg5cqTZZTmlc+fOMWnSJObPn8/JkyepXr06gwcP5plnnsHDw8Ps8sotBSARERFxOhoDJCIiIk5HAUhEREScjgKQiIiIOB0FIBEREXE6CkAiIiLidBSARERExOkoAImIiIjTUQASEcmH5cuXY7FYOHv2rNmliEgxUAASERERp6MAJCIiIk5HAUhEygWbzca0adOIiorC29ubFi1a8O233wIXu6cWLlxI8+bN8fLyokOHDmzbti3HNb777juaNGmCp6cnkZGRvP766zleT09P56mnniIiIgJPT0/q1q3Lhx9+mOOY9evX07ZtW3x8fOjUqVOuVdNFpHxQABKRcmHatGl8+umnzJo1i+3btzN69GjuuusuVqxY4TjmiSee4PXXX2fdunWEhITQt29fMjMzAXtwGTBgAIMGDWLr1q08++yzTJo0iY8//thx/tChQ/nyyy9588032blzJ++99x5+fn456pgwYQKvv/46f//9N25ublqRW6Sc0mKoIlLmpaenU7lyZZYuXUrHjh0d+++77z5SU1O5//77ufbaa/nqq68YOHAgAKdPn6ZGjRp8/PHHDBgwgCFDhnDq1Cl++eUXx/lPPvkkCxcuZPv27ezZs4cGDRqwZMkSevTokauG5cuXc+2117J06VKuv/56AH7++Wduvvlmzp8/j5eXVwn/FESkOKkFSETKvH379pGamsoNN9yAn5+f4/Hpp58SExPjOO7ScFS5cmUaNGjAzp07Adi5cyedO3fOcd3OnTuzd+9erFYrmzZtwtXVlWuuueaKtTRv3tyxXa1aNQBOnjxZ5M8oIqXLzewCRESuJjk5GYCFCxcSHh6e4zVPT88cIaiwvL2983Wcu7u7Y9tisQD28UkiUr6oBUhEyrzGjRvj6enJ4cOHqVu3bo5HRESE47i//vrLsX3mzBn27NlDo0aNAGjUqBGrVq3Kcd1Vq1ZRv359XF1dadasGTabLceYIhGpuNQCJCJlnr+/P2PHjmX06NHYbDa6dOlCYmIiq1atIiAggFq1agEwZcoUqlSpQmhoKBMmTCA4OJhbb70VgP/85z+0a9eOqVOnMnDgQFavXs3MmTN55513AIiMjGTYsGHcc889vPnmm7Ro0YJDhw5x8uRJBgwYYNZHF5ESogAkIuXC1KlTCQkJYdq0aezfv5+goCBat27N008/7eiCeumll3jsscfYu3cvLVu25Mcff8TDwwOA1q1b8/XXX/PMM88wdepUqlWrxpQpUxg+fLjjPd59912efvppHnroIRISEqhZsyZPP/20GR9XREqYZoGJSLmXPUPrzJkzBAUFmV2OiJQDGgMkIiIiTkcBSERERJyOusBERETE6agFSERERJyOApCIiIg4HQUgERERcToKQCIiIuJ0FIBERETE6SgAiYiIiNNRABIRERGnowAkIiIiTkcBSERERJzO/wNC4kPMSUdYyQAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.plot(history.history['loss'])\n",
        "plt.plot(history.history['val_loss'])\n",
        "\n",
        "plt.title('model loss')\n",
        "plt.ylabel('loss')\n",
        "plt.xlabel('epoch')\n",
        "\n",
        "plt.legend(['training data', 'validation data'], loc = 'upper right')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 489
        },
        "id": "8AWbiN6JKqLp",
        "outputId": "4acc7e86-1055-4daf-bef7-866acd7b5353"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.legend.Legend at 0x798b500dcce0>"
            ]
          },
          "metadata": {},
          "execution_count": 78
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkAAAAHHCAYAAABXx+fLAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAdWNJREFUeJzt3Xd4FNXbxvHv7qZ3QjoEEnon1BiKokaDhWIDK0WFV6wY0R9YQEDFLiIoigXsWFBRFIQoCEgTpPcSahJIIB1Sdvf9Y2ExEhBIsptyf65rL3dnZ+Y8A0huzsw5x2C1Wq2IiIiI1CBGZxcgIiIi4mgKQCIiIlLjKACJiIhIjaMAJCIiIjWOApCIiIjUOApAIiIiUuMoAImIiEiNowAkIiIiNY4CkIiIiNQ4CkAiUuUlJydjMBiYPn36BR+7cOFCDAYDCxcuPOd+06dPx2AwkJycfFE1ikjlogAkIiIiNY4CkIiIiNQ4CkAiIiJS4ygAiUiZPfvssxgMBrZv386dd96Jv78/wcHBPPPMM1itVvbv30+fPn3w8/MjLCyM11577YxzHD58mHvuuYfQ0FA8PDxo27YtM2bMOGO/zMxMBg0ahL+/PwEBAQwcOJDMzMxS69q6dSs333wzgYGBeHh40LFjR2bPnl2u1/7222/TsmVL3N3diYiI4IEHHjijnh07dnDTTTcRFhaGh4cHdevW5dZbbyUrK8u+z/z58+nWrRsBAQH4+PjQtGlTnnzyyXKtVUROc3F2ASJSffTv35/mzZvz4osvMmfOHJ577jkCAwN59913ueKKK3jppZf47LPPGDFiBJ06deLSSy8F4Pjx4/To0YOdO3fy4IMPEh0dzddff82gQYPIzMzkkUceAcBqtdKnTx+WLFnCfffdR/Pmzfnuu+8YOHDgGbVs2rSJrl27UqdOHUaOHIm3tzdfffUVffv25dtvv+WGG24o8/U+++yzjB07lvj4eIYNG8a2bdt45513WLVqFUuXLsXV1ZXCwkISEhIoKCjgoYceIiwsjIMHD/LTTz+RmZmJv78/mzZt4vrrr6dNmzaMGzcOd3d3du7cydKlS8tco4ichVVEpIzGjBljBaxDhw61bysuLrbWrVvXajAYrC+++KJ9+7Fjx6yenp7WgQMH2rdNnDjRClg//fRT+7bCwkJrXFyc1cfHx5qdnW21Wq3W77//3gpYX3755RLtdO/e3QpYP/roI/v2K6+80tq6dWvriRMn7NssFou1S5cu1saNG9u3/f7771bA+vvvv5/zGj/66CMrYN2zZ4/VarVaDx8+bHVzc7NeffXVVrPZbN9v8uTJVsD64YcfWq1Wq/Xvv/+2Atavv/76rOd+4403rID1yJEj56xBRMqPboGJSLm599577e9NJhMdO3bEarVyzz332LcHBATQtGlTdu/ebd/2888/ExYWxm233Wbf5urqysMPP0xubi6LFi2y7+fi4sKwYcNKtPPQQw+VqOPo0aP89ttv9OvXj5ycHNLT00lPTycjI4OEhAR27NjBwYMHy3StCxYsoLCwkOHDh2M0nv6rdMiQIfj5+TFnzhwA/P39AZg3bx75+fmlnisgIACAH374AYvFUqa6ROT8KACJSLmpV69eic/+/v54eHgQFBR0xvZjx47ZP+/du5fGjRuXCBIAzZs3t39/6r/h4eH4+PiU2K9p06YlPu/cuROr1cozzzxDcHBwideYMWMA2zNHZXGqpn+37ebmRoMGDezfR0dHk5iYyPvvv09QUBAJCQlMmTKlxPM//fv3p2vXrtx7772EhoZy66238tVXXykMiVQgPQMkIuXGZDKd1zawPc9TUU4FhxEjRpCQkFDqPo0aNaqw9v/ttddeY9CgQfzwww/8+uuvPPzww0yYMIHly5dTt25dPD09+eOPP/j999+ZM2cOc+fOZebMmVxxxRX8+uuvZ/01FJGLpx4gEXG6+vXrs2PHjjN6PLZu3Wr//tR/U1JSyM3NLbHftm3bSnxu0KABYLuNFh8fX+rL19e3zDWX1nZhYSF79uyxf39K69atefrpp/njjz9YvHgxBw8eZOrUqfbvjUYjV155Ja+//jqbN2/m+eef57fffuP3338vU50iUjoFIBFxumuvvZbU1FRmzpxp31ZcXMxbb72Fj48Pl112mX2/4uJi3nnnHft+ZrOZt956q8T5QkJC6NGjB++++y4pKSlntHfkyJEy1xwfH4+bmxuTJk0q0Zv1wQcfkJWVxXXXXQdAdnY2xcXFJY5t3bo1RqORgoICwPbM0r/FxMQA2PcRkfKlW2Ai4nRDhw7l3XffZdCgQaxevZqoqCi++eYbli5dysSJE+29Nb169aJr166MHDmS5ORkWrRowaxZs0o8T3PKlClT6NatG61bt2bIkCE0aNCA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PPjgg3z33Xf89ttvREdHO7ukGu3KK69kw4YNrF271v7q2LEjd9xxB2vXrlX4cYKuXbueMTXE9u3bqV+/vpMqqtny8/MxGkv+uDaZTFgsFidVVD2oB8hBEhMTGThwIB07dqRz585MnDiRvLw8Bg8e7OzSapwHHniAzz//nB9++AFfX19SU1MB8Pf3x9PT08nV1Ty+vr5nPH/l7e1N7dq19VyWkzz66KN06dKFF154gX79+rFy5Uree+893nvvPWeXViP16tWL559/nnr16tGyZUv+/vtvXn/9de6++25nl1alaRi8A02ePJlXXnmF1NRUYmJimDRpErGxsc4uq8YxGAylbv/oo48YNGiQY4uRUvXo0UPD4J3sp59+YtSoUezYsYPo6GgSExMZMmSIs8uqkXJycnjmmWf47rvvOHz4MBEREdx2222MHj0aNzc3Z5dXZSkAiYiISI2jZ4BERESkxlEAEhERkRpHAUhERERqHAUgERERqXEUgERERKTGUQASERGRGkcBSERERGocBSARkfOwcOFCDAYDmZmZzi5FRMqBApCIiIjUOApAIiIiUuMoAIlIlWCxWJgwYQLR0dF4enrStm1bvvnmG+D07ak5c+bQpk0bPDw8uOSSS9i4cWOJc3z77be0bNkSd3d3oqKieO2110p8X1BQwP/+9z8iIyNxd3enUaNGfPDBByX2Wb16NR07dsTLy4suXbqcsWq6iFQNCkAiUiVMmDCBjz/+mKlTp7Jp0yYeffRR7rzzThYtWmTf5/HHH+e1115j1apVBAcH06tXL4qKigBbcOnXrx+33norGzZs4Nlnn+WZZ55h+vTp9uMHDBjAF198waRJk9iyZQvvvvsuPj4+Jep46qmneO211/jrr79wcXHRitwiVZQWQxWRSq+goIDAwEAWLFhAXFycffu9995Lfn4+Q4cO5fLLL+fLL7+kf//+ABw9epS6desyffp0+vXrxx133MGRI0f49ddf7cc/8cQTzJkzh02bNrF9+3aaNm3K/PnziY+PP6OGhQsXcvnll7NgwQKuvPJKAH7++Weuu+46jh8/joeHRwX/KohIeVIPkIhUejt37iQ/P5+rrroKHx8f++vjjz9m165d9v3+GY4CAwNp2rQpW7ZsAWDLli107dq1xHm7du3Kjh07MJvNrF27FpPJxGWXXXbOWtq0aWN/Hx4eDsDhw4fLfI0i4lguzi5AROS/5ObmAjBnzhzq1KlT4jt3d/cSIehieXp6ntd+rq6u9vcGgwGwPZ8kIlWLeoBEpNJr0aIF7u7u7Nu3j0aNGpV4RUZG2vdbvny5/f2xY8fYvn07zZs3B6B58+YsXbq0xHmXLl1KkyZNMJlMtG7dGovFUuKZIhGpvtQDJCKVnq+vLyNGjODRRx/FYrHQrVs3srKyWLp0KX5+ftSvXx+AcePGUbt2bUJDQ3nqqacICgqib9++ADz22GN06tSJ8ePH079/f5YtW8bkyZN5++23AYiKimLgwIHcfffdTJo0ibZt27J3714OHz5Mv379nHXpIlJBFIBEpEoYP348wcHBTJgwgd27dxMQEED79u158skn7begXnzxRR555BF27NhBTEwMP/74I25ubgC0b9+er776itGjRzN+/HjCw8MZN24cgwYNsrfxzjvv8OSTT3L//feTkZFBvXr1ePLJJ51xuSJSwTQKTESqvFMjtI4dO0ZAQICzyxGRKkDPAImIiEiNowAkIiIiNY5ugYmIiEiNox4gERERqXEUgERERKTGUQASERGRGkcBSERERGocBSARERGpcRSAREREpMZRABIREZEaRwFIREREahwFIBEREalx/h9naQLRrq3rRAAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "loss, accuracy = model.evaluate(X_test_std, Y_test)\n",
        "print(accuracy)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "d59pWLRKK6wW",
        "outputId": "aac35b76-eb57-48af-cd69-bf6fc8129ce6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[1m4/4\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 10ms/step - accuracy: 0.9616 - loss: 0.1393\n",
            "0.9561403393745422\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(X_test_std.shape)\n",
        "print(X_test_std[0])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SWm4LbQpK_yt",
        "outputId": "a681d669-0b17-4f17-de85-e93bf67ce3d9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(114, 30)\n",
            "[-0.04462793 -1.41612656 -0.05903514 -0.16234067  2.0202457  -0.11323672\n",
            "  0.18500609  0.47102419  0.63336386  0.26335737  0.53209124  2.62763999\n",
            "  0.62351167  0.11405261  1.01246781  0.41126289  0.63848593  2.88971815\n",
            " -0.41675911  0.74270853 -0.32983699 -1.67435595 -0.36854552 -0.38767294\n",
            "  0.32655007 -0.74858917 -0.54689089 -0.18278004 -1.23064515 -0.6268286 ]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "Y_pred = model.predict(X_test_std)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ot_-ZGpBLH1P",
        "outputId": "802898eb-33a9-41a8-f318-6702e9cc632b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[1m4/4\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 16ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(Y_pred.shape)\n",
        "print(Y_pred[0])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3WB5SJSuLM2Z",
        "outputId": "a2bfc190-2307-41d1-ecec-b21162467e46"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(114, 2)\n",
            "[0.16299675 0.5231131 ]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(X_test_std)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MwP1AjK8LPGf",
        "outputId": "f87ce490-7c1e-4abc-9135-b0b705b54888"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[-0.04462793 -1.41612656 -0.05903514 ... -0.18278004 -1.23064515\n",
            "  -0.6268286 ]\n",
            " [ 0.24583601 -0.06219797  0.21802678 ...  0.54129749  0.11047691\n",
            "   0.0483572 ]\n",
            " [-1.26115925 -0.29051645 -1.26499659 ... -1.35138617  0.269338\n",
            "  -0.28231213]\n",
            " ...\n",
            " [ 0.72709489  0.45836817  0.75277276 ...  1.46701686  1.19909344\n",
            "   0.65319961]\n",
            " [ 0.25437907  1.33054477  0.15659489 ... -1.29043534 -2.22561725\n",
            "  -1.59557344]\n",
            " [ 0.84100232 -0.06676434  0.8929529  ...  2.15137705  0.35629355\n",
            "   0.37459546]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(Y_pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7ZasSf3ULVY2",
        "outputId": "7130928f-0dbb-4f83-8710-73200f9f98c7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[1.62996754e-01 5.23113072e-01]\n",
            " [3.73348981e-01 4.18467015e-01]\n",
            " [7.85766691e-02 9.18937743e-01]\n",
            " [9.99017537e-01 7.61979667e-04]\n",
            " [4.60914612e-01 4.75711256e-01]\n",
            " [9.85900939e-01 9.61848721e-03]\n",
            " [2.60063559e-01 7.96142936e-01]\n",
            " [6.75582290e-02 8.99281383e-01]\n",
            " [1.39661431e-01 7.71863937e-01]\n",
            " [8.77799168e-02 7.81208754e-01]\n",
            " [4.07810390e-01 5.85940838e-01]\n",
            " [1.69116050e-01 7.01455116e-01]\n",
            " [8.96506310e-02 7.60775149e-01]\n",
            " [1.64936036e-01 6.33978128e-01]\n",
            " [8.69391561e-02 8.92043471e-01]\n",
            " [8.82034957e-01 2.22910345e-01]\n",
            " [7.36231059e-02 8.81814301e-01]\n",
            " [1.85802266e-01 7.84808099e-01]\n",
            " [7.74384961e-02 8.48431110e-01]\n",
            " [9.00534928e-01 6.75926916e-03]\n",
            " [6.15049183e-01 7.62663186e-01]\n",
            " [1.17512949e-01 8.53065908e-01]\n",
            " [1.40882671e-01 8.27967346e-01]\n",
            " [4.90039773e-02 8.43618631e-01]\n",
            " [1.81102082e-01 7.19568312e-01]\n",
            " [9.59899008e-01 2.87188329e-02]\n",
            " [2.20269248e-01 6.30295694e-01]\n",
            " [3.20121109e-01 5.34191310e-01]\n",
            " [9.42060232e-01 8.08826536e-02]\n",
            " [9.60898399e-01 6.97184652e-02]\n",
            " [1.70709923e-01 7.21883118e-01]\n",
            " [1.11122049e-01 7.59488165e-01]\n",
            " [1.88613206e-01 8.13662946e-01]\n",
            " [9.99031663e-01 1.06006302e-03]\n",
            " [9.73549664e-01 8.73636641e-03]\n",
            " [2.58801311e-01 7.46016085e-01]\n",
            " [4.71296348e-02 9.24997509e-01]\n",
            " [1.84996501e-01 6.99407160e-01]\n",
            " [4.70325164e-02 8.31314445e-01]\n",
            " [1.10291429e-01 7.46694565e-01]\n",
            " [9.94165838e-01 3.53542011e-04]\n",
            " [8.05035114e-01 2.29943261e-01]\n",
            " [3.12198978e-02 8.23159397e-01]\n",
            " [2.14948997e-01 8.50643635e-01]\n",
            " [5.40957987e-01 9.15931910e-02]\n",
            " [1.37461960e-01 8.42701316e-01]\n",
            " [6.73962831e-02 8.76981258e-01]\n",
            " [5.63435592e-02 9.13433015e-01]\n",
            " [9.94609594e-01 6.17604516e-03]\n",
            " [9.17853892e-01 5.29843904e-02]\n",
            " [1.29228026e-01 7.53153741e-01]\n",
            " [7.58854687e-01 2.13257194e-01]\n",
            " [3.88341039e-01 4.70478714e-01]\n",
            " [6.60795569e-02 8.77355874e-01]\n",
            " [7.63261020e-02 8.88982594e-01]\n",
            " [4.15536314e-01 3.58389705e-01]\n",
            " [5.99919669e-02 6.63273871e-01]\n",
            " [6.47338182e-02 9.34146762e-01]\n",
            " [1.89684421e-01 6.56974390e-02]\n",
            " [2.90238887e-01 7.91163743e-01]\n",
            " [2.49021053e-01 6.17768347e-01]\n",
            " [7.66371787e-01 4.71500196e-02]\n",
            " [5.92951439e-02 8.48762035e-01]\n",
            " [9.45097268e-01 1.59827881e-02]\n",
            " [8.87199998e-01 1.39304221e-01]\n",
            " [4.69239265e-01 6.02796435e-01]\n",
            " [9.90034699e-01 1.54132834e-02]\n",
            " [9.23829377e-01 6.75053671e-02]\n",
            " [3.31918806e-01 5.62586248e-01]\n",
            " [3.28072429e-01 5.26386440e-01]\n",
            " [7.35574126e-01 2.72875667e-01]\n",
            " [8.64878714e-01 7.31342807e-02]\n",
            " [1.27326950e-01 8.12912107e-01]\n",
            " [7.31436014e-01 1.14149928e-01]\n",
            " [3.79407555e-02 9.02726591e-01]\n",
            " [6.71147883e-01 9.30984989e-02]\n",
            " [1.01130560e-01 8.26571584e-01]\n",
            " [8.24834704e-02 8.04333985e-01]\n",
            " [1.65699303e-01 6.20957136e-01]\n",
            " [4.77302521e-01 1.66441888e-01]\n",
            " [9.56957757e-01 4.40356508e-02]\n",
            " [7.99175203e-01 2.02225968e-01]\n",
            " [8.41607869e-01 1.99155286e-02]\n",
            " [3.59325022e-01 6.99471831e-01]\n",
            " [1.66088328e-01 7.29631543e-01]\n",
            " [5.47134817e-01 4.54844505e-01]\n",
            " [2.97659487e-01 8.27088535e-01]\n",
            " [1.67020336e-01 7.78998852e-01]\n",
            " [3.48458171e-01 5.95931351e-01]\n",
            " [9.65181112e-01 5.56348637e-02]\n",
            " [1.48157418e-01 8.03878486e-01]\n",
            " [1.54484078e-01 7.26959348e-01]\n",
            " [4.02739495e-01 7.77419567e-01]\n",
            " [9.62291062e-01 1.36189923e-01]\n",
            " [7.16698647e-01 2.00572357e-01]\n",
            " [1.49938211e-01 6.53751373e-01]\n",
            " [9.76074696e-01 1.88615602e-02]\n",
            " [8.11300159e-01 4.67555970e-02]\n",
            " [1.02994174e-01 7.45044708e-01]\n",
            " [6.56563938e-02 9.24822986e-01]\n",
            " [4.34071571e-02 9.30820644e-01]\n",
            " [5.55514514e-01 1.19498722e-01]\n",
            " [9.95957792e-01 9.47208144e-03]\n",
            " [9.91240263e-01 1.88663194e-03]\n",
            " [1.46347776e-01 7.81819344e-01]\n",
            " [1.82252049e-01 8.43989730e-01]\n",
            " [1.53370574e-01 9.08259153e-01]\n",
            " [2.85607845e-01 8.48232925e-01]\n",
            " [6.72030151e-02 9.70829129e-01]\n",
            " [1.64166510e-01 7.00972915e-01]\n",
            " [9.89874005e-01 1.97010376e-02]\n",
            " [9.68095899e-01 9.03610047e-03]\n",
            " [1.63969591e-01 4.35511202e-01]\n",
            " [8.43385696e-01 2.17308514e-02]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#  argmax function\n",
        "\n",
        "my_list = [0.25, 0.56]\n",
        "\n",
        "index_of_max_value = np.argmax(my_list)\n",
        "print(my_list)\n",
        "print(index_of_max_value)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "45iZqKtYLc76",
        "outputId": "d8a362f5-4d62-4966-e180-00ba5d3f01c3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0.25, 0.56]\n",
            "1\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# converting the prediction probability to class labels\n",
        "\n",
        "Y_pred_labels = [np.argmax(i) for i in Y_pred]\n",
        "print(Y_pred_labels)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TuCyjRQULujf",
        "outputId": "f4492fa6-f68b-49dc-b84f-4940c041cf9d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[np.int64(1), np.int64(1), np.int64(1), np.int64(0), np.int64(1), np.int64(0), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(0), np.int64(1), np.int64(1), np.int64(1), np.int64(0), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(0), np.int64(1), np.int64(1), np.int64(0), np.int64(0), np.int64(1), np.int64(1), np.int64(1), np.int64(0), np.int64(0), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(0), np.int64(0), np.int64(1), np.int64(1), np.int64(0), np.int64(1), np.int64(1), np.int64(1), np.int64(0), np.int64(0), np.int64(1), np.int64(0), np.int64(1), np.int64(1), np.int64(1), np.int64(0), np.int64(1), np.int64(1), np.int64(0), np.int64(1), np.int64(1), np.int64(0), np.int64(1), np.int64(0), np.int64(0), np.int64(1), np.int64(0), np.int64(0), np.int64(1), np.int64(1), np.int64(0), np.int64(0), np.int64(1), np.int64(0), np.int64(1), np.int64(0), np.int64(1), np.int64(1), np.int64(1), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(1), np.int64(1), np.int64(0), np.int64(1), np.int64(1), np.int64(1), np.int64(0), np.int64(1), np.int64(1), np.int64(1), np.int64(0), np.int64(0), np.int64(1), np.int64(0), np.int64(0), np.int64(1), np.int64(1), np.int64(1), np.int64(0), np.int64(0), np.int64(0), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(0), np.int64(0), np.int64(1), np.int64(0)]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "input_data = (12.768,34,74.72,427.9,0.09637,0.0466,0.01657,0.0111,0.1495,0.05888,0.4062,1.21,2.635,28.47,0.005857,0.009758,0.168,0.045,0.02406,0.001769,12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)\n",
        "\n",
        "# change the input_data to a numpy array\n",
        "input_data_as_numpy_array = np.asarray(input_data)\n",
        "\n",
        "# reshape the numpy array as we are predicting for one data point\n",
        "input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)\n",
        "\n",
        "# standardizing the input data\n",
        "input_data_std = scaler.transform(input_data_reshaped)\n",
        "\n",
        "prediction = model.predict(input_data_std)\n",
        "print(prediction)\n",
        "\n",
        "prediction_label = [np.argmax(prediction)]\n",
        "if(prediction_label[0] == 0):\n",
        "  print('The tumor is Malignant')\n",
        "\n",
        "else:\n",
        "  print('The tumor is Benign')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jJsceTj2L9hE",
        "outputId": "df1ae55a-d79a-47d3-d066-68a298bd4764"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 38ms/step\n",
            "[[0.06799073 0.69747066]]\n",
            "The tumor is Benign\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names\n",
            "  warnings.warn(\n"
          ]
        }
      ]
    }
  ]
}
