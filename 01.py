import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def generate_features(n_sapmples=1000, student_id=123456):

    np.random.seed(student_id%1000)

    # feature 1: normal distribution data 
    feature_1 = np.random.normal(loc=50, scale=15, size=n_sapmples)

    # feature 2: Exponential distribution data (highly skewed)
    feature_2 = np.random.exponential(scale=2, size=n_sapmples)

    # Include a few extream artificial outliers in feature 2
    feature_2[np.random.choice(n_sapmples, 5, replace=False)] = np.random.uniform(50, 100, 5)

    return feature_1, feature_2


student_id = 230
f1, f2 = generate_features(n_sapmples=1000, student_id=student_id)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(f1, bins=50, color='blue', alpha=0.7) # histogram for feature 1
plt.title('Feature 1 Distribution')
plt.subplot(1, 2, 2)
plt.hist(f2, bins=50, color='orange', alpha=0.7) # histogram for feature 2
plt.title('Feature 2 Distribution')
plt.show()


######################################################################
##              Sample Standardization of Features                  ##
######################################################################
standard_scaler = StandardScaler()
f1_scaled = standard_scaler.fit_transform(f1.reshape(-1, 1)).flatten()
f2_scaled = standard_scaler.fit_transform(f2.reshape(-1, 1)).flatten()


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(f1_scaled, bins=50, color='blue', alpha=0.7) # histogram for feature 1
plt.title('Feature 1 Scaled Distribution')
plt.subplot(1, 2, 2)
plt.hist(f2_scaled, bins=50, color='orange', alpha=0.7) # histogram for feature 2
plt.title('Feature 2 Scaled Distribution')
plt.show()




#######################################################################
##        Non Linear Transformations  - Log Transformation           ##
#######################################################################


f2_log   = np.log1p(f2)  # log(1 + x) to handle zero values since feature 2  values are close to zero
f2_scaled_log = standard_scaler.fit_transform(f2_log.reshape(-1, 1)).flatten()


plt.subplot(1,2,1)
plt.hist(f2_log, bins=50, color='green', alpha=0.7)
plt.title('Feature 2 after Log Transformation')
plt.xlabel('log(1 + x)')
plt.ylabel('Frequency')

plt.subplot(1,2,2)
plt.hist(f2_scaled_log, bins=50, color='purple', alpha=0.7)
plt.title('Feature 2 after Log + Standard Scaling')
plt.xlabel('Scaled Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()





