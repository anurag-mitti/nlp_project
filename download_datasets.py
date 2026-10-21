import kagglehub

amazon_path = kagglehub.dataset_download("grikomsn/amazon-cell-phones-reviews")
semeval_path = kagglehub.dataset_download("charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis")

print(amazon_path)
print(semeval_path)
