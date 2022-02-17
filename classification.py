from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import joblib

X = joblib.load('X.joblib')
y = joblib.load('y.joblib')

vectorizer = CountVectorizer(max_features=1500, min_df=10, max_df=0.8, stop_words='english')
X = vectorizer.fit_transform(X).toarray()


transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



classifier_log_reg = LogisticRegression()
classifier_log_reg.fit(X_train, y_train)

#classifier_k_neigh = KNeighborsClassifier(5)
#classifier_k_neigh.fit(X_train, y_train)

#classifier_naive_b = GaussianNB()
#classifier_naive_b.fit(X_train, y_train)

#classifier_svc = make_pipeline(StandardScaler(), SVC())
#classifier_svc.fit(X_train, y_train)

#classifier_rand_forest = RandomForestClassifier(n_estimators = 60, max_depth = 150 )
#classifier_rand_forest.fit(X_train, y_train)

#classifier_mlp_neur_net = MLPClassifier(random_state=1, max_iter=100, activation= 'logistic', alpha=0.001, learning_rate='adaptive')
#classifier_mlp_neur_net.fit(X_train, y_train)


y_pred = classifier_log_reg.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
