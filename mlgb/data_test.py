from mlgb.data import get_multitask_label_data
print("=====")


feature_names, (x_train, y_train), (x_test, y_test) = get_multitask_label_data(
        n_samples=100,
        negative_class_weight=0.9,
        multitask_cvr=0.5,
        test_size=0.15,
        seed=0,
    )
# (dense, sparse, seq)
# dense: (batch_size, numerical_fea_num)
# sparse: (batch_size, sparse_fea_num)) , sparse_fea_num 实际包含 pure_sparse_size + numerical_fea_size (bin)
# seq: (batch_size, seq_fea_num, seq_length)

index = 2
print(feature_names[index])
print(x_train[index].shape)
print(x_train[index])