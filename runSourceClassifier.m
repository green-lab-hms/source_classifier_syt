function runSourceClassifier(sourceMat, labelsCsv, convNetFn)

addpath('/home/jg319/code/source_classifier_syt/');
load(sourceMat);
l = clusterSourcesWithCurrentNn(A, convNetFn);
csvwrite(labelsCsv, l);