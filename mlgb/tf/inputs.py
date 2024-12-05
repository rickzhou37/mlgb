# coding=utf-8
# author=uliontse

"""
Copyright 2020 UlionTse

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from mlgb.tf.configs import (
    tf, 
    Flatten,
    InputsModeList,
)
from mlgb.tf.functions import (
    SparseEmbeddingLayer,
    DenseEmbeddingLayer,
    MultiValuedPoolingLayer,
)
from mlgb.error import MLGBError


__all__ = ['InputsLayer']


class FeatureNamesLayer:
    def __init__(self, inputs_if_multivalued=False, inputs_if_sequential=False):
        self.inputs_if_multivalued = inputs_if_multivalued # 是否使用 mv feature, 不使用为 None（在第三位置）
        self.inputs_if_sequential = inputs_if_sequential # 是否使用 seq feature, 不使用为 None（在第四位置）

    def check(self, inputs):
        self.inputs_num = len(inputs)

        if self.inputs_num not in (2, 3, 4):
            raise MLGBError
        if self.inputs_num == 2 and (self.inputs_if_multivalued or self.inputs_if_sequential):
            raise MLGBError
        if self.inputs_num == 3 and (self.inputs_if_multivalued and self.inputs_if_sequential):
            raise MLGBError
        return
    # （dense_feature_names, sparse_feature_names, mv_feature_names, seq_feature_names）
    def get(self, inputs):
        self.check(inputs)
        dense_feature_names, sparse_feature_names = inputs[0], inputs[1]

        mv_feature_names, seq_feature_names = None, None
        if self.inputs_num in (3, 4):
            if self.inputs_if_multivalued:
                mv_feature_names = inputs[2]
            if self.inputs_if_sequential:
                seq_feature_names = inputs[-1]
        
        return dense_feature_names, sparse_feature_names, mv_feature_names, seq_feature_names

class SequentialInputsLayer(tf.keras.layers.Layer):
    def __init__(self, inputs_if_multivalued=False, inputs_if_sequential=False,
                 inputs_if_embed_dense=False, outputs_dense_if_add_sparse=True,
                 embed_sparse_cate_fn=None, embed_sparse_mv_fn=None, embed_sparse_seq_fn=None,
                 embed_dense_fn=None, pool_mv_fn=None, pool_seq_fn=None):
        super().__init__()
        self.inputs_if_multivalued = inputs_if_multivalued
        self.inputs_if_sequential = inputs_if_sequential
        self.inputs_if_embed_dense = inputs_if_embed_dense
        self.outputs_dense_if_add_sparse = outputs_dense_if_add_sparse
        self.embed_sparse_cate_fn = embed_sparse_cate_fn
        self.embed_sparse_mv_fn = embed_sparse_mv_fn
        self.embed_sparse_seq_fn = embed_sparse_seq_fn
        self.embed_dense_fn = embed_dense_fn
        self.pool_mv_fn = pool_mv_fn
        self.pool_seq_fn = pool_seq_fn
        self.flatten_fn = Flatten()

    def build(self, input_shape):
        # [dens_input, sparse_input, mv_input, seq_input]
        self.inputs_num = len(input_shape)
        
        if self.inputs_num not in (2, 3, 4):
            raise MLGBError
        if self.inputs_num == 2 and (self.inputs_if_multivalued or self.inputs_if_sequential):
            raise MLGBError
        if self.inputs_num == 3 and (self.inputs_if_multivalued and self.inputs_if_sequential):
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):  # torch, tf216:Tuple[numpy.ndarray, ...], tf210: Tuple[Tensor, ...].
        # [dens_input, sparse_input, mv_input, seq_input]
        dense_2d_tensor = tf.convert_to_tensor(inputs[0], dtype=tf.float32) # (batch_size, dense_fea_num)
        sparse_2d_tensor = tf.convert_to_tensor(inputs[1], dtype=tf.int32)  # (batch_size, sparse_fea_num)
        embed_cate_2d_tensor, embed_cate_3d_tensor = self.embed_sparse_cate_fn(sparse_2d_tensor)
        print("#SequentialInputLayer 1, dense_input shape", dense_2d_tensor.shape)
        print("#SequentialInputLayer 2, sparse_input shape", sparse_2d_tensor.shape)
        print("#SequentialInputLayer 3, embed_cate_2d_tensor shape", embed_cate_2d_tensor.shape)
        print("#SequentialInputLayer 4, embed_cate_3d_tensor shape", embed_cate_3d_tensor.shape)

        if self.inputs_if_embed_dense and embed_cate_3d_tensor.shape.rank != 3:
            raise MLGBError

        if self.inputs_if_embed_dense and embed_cate_3d_tensor.shape.rank == 3:
            print("#SequentialInputsLayer 5, use embed_dense")
            # [batch, numerical_fea_num, emb_dim]
            embed_dense_3d_tensor = self.embed_dense_fn(dense_2d_tensor)
            # [batch, sparse_fea_num + numerical_fea_num, emb_dim ]
            embed_cate_3d_tensor = tf.concat([embed_cate_3d_tensor, embed_dense_3d_tensor], axis=1)
            print("#SequentialInputsLayer 6, embed_dense_3d_tensor shape", embed_dense_3d_tensor.shape)
            print("#SequentialInputsLayer 7, embed_cate_3d_tensor shape", embed_cate_3d_tensor.shape)

        if self.outputs_dense_if_add_sparse:
            print("#SequentialInputsLayer 8, use outputs_dense_add_sparse")
            dense_2d_tensor = tf.concat([dense_2d_tensor, embed_cate_2d_tensor], axis=1)
            print("#SequentialInputsLayer 9, dense_2d_tensor shape", dense_2d_tensor.shape)

        seq_3d_tensor = None
        if self.inputs_num in (3, 4):
            if self.inputs_if_multivalued:
                print("#SequentialInputsLayer 10, use multivalued")
                mv_tensor = tf.convert_to_tensor(inputs[2], dtype=tf.int32)
                _, embed_mv_4d_tensor = self.embed_sparse_mv_fn(mv_tensor)
                mv_3d_tensor = self.pool_mv_fn(embed_mv_4d_tensor)
                mv_2d_tensor = self.flatten_fn(mv_3d_tensor)

                if embed_cate_3d_tensor.shape.rank == 2:
                    embed_cate_3d_tensor = tf.concat([embed_cate_3d_tensor, mv_2d_tensor], axis=1)
                else:
                    embed_cate_3d_tensor = tf.concat([embed_cate_3d_tensor, mv_3d_tensor], axis=1)

                if self.outputs_dense_if_add_sparse:
                    dense_2d_tensor = tf.concat([dense_2d_tensor, mv_2d_tensor], axis=1)

            if self.inputs_if_sequential:
                print("#SequentialInputsLayer 11, use sequential")
                seq_tensor = tf.convert_to_tensor(inputs[-1], dtype=tf.int32)  #
                _, embed_seq_4d_tensor = self.embed_sparse_seq_fn(seq_tensor)
                seq_3d_tensor = self.pool_seq_fn(embed_seq_4d_tensor)

        return dense_2d_tensor, embed_cate_3d_tensor, seq_3d_tensor

class FeatureInputsLayer(tf.keras.layers.Layer):
    def __init__(self, inputs_if_multivalued=False, inputs_if_sequential=False,
                 inputs_if_embed_dense=False, outputs_dense_if_add_sparse=True,
                 embed_sparse_cate_fn=None, embed_sparse_mv_fn=None, embed_sparse_seq_fn=None,
                 embed_dense_fn=None, pool_mv_fn=None, pool_seq_fn=None):
        super().__init__()
        self.inputs_if_sequential = inputs_if_sequential
        self.outputs_dense_if_add_sparse = outputs_dense_if_add_sparse
        
        self.inputs_seq_fn = SequentialInputsLayer(
            inputs_if_multivalued=inputs_if_multivalued,
            inputs_if_sequential=inputs_if_sequential,
            inputs_if_embed_dense=inputs_if_embed_dense,
            outputs_dense_if_add_sparse=outputs_dense_if_add_sparse,
            embed_sparse_cate_fn=embed_sparse_cate_fn,
            embed_sparse_mv_fn=embed_sparse_mv_fn,
            embed_sparse_seq_fn=embed_sparse_seq_fn,
            embed_dense_fn=embed_dense_fn,
            pool_mv_fn=pool_mv_fn,
            pool_seq_fn=pool_seq_fn,
        )
        self.flatten_fn = Flatten()

    def build(self, input_shape):
        # [dens_input, sparse_input, mv_input, seq_input] （后俩个可选）
        if len(input_shape) not in (2, 3, 4):
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        dense_2d_tensor, embed_cate_3d_tensor, seq_3d_tensor = self.inputs_seq_fn(inputs)
        print("#FeatureInputsLayer 1, dense_2d_tensor shape", dense_2d_tensor.shape)
        print("#FeatureInputsLayer 2, embed_cate_3d_tensor shape", embed_cate_3d_tensor.shape)
        
        if self.inputs_if_sequential:
            print("#FeatureInputsLayer 3, use input_sequential")
            seq_2d_tensor = self.flatten_fn(seq_3d_tensor)

            if embed_cate_3d_tensor.shape.rank == 2:
                embed_cate_3d_tensor = tf.concat([embed_cate_3d_tensor, seq_2d_tensor], axis=1)
            else:
                embed_cate_3d_tensor = tf.concat([embed_cate_3d_tensor, seq_3d_tensor], axis=1)

            if self.outputs_dense_if_add_sparse:
                dense_2d_tensor = tf.concat([dense_2d_tensor, seq_2d_tensor], axis=1)
        return dense_2d_tensor, embed_cate_3d_tensor

# 入口1
# inputs_if_multivalued 是否使用 mv 特征
# inputs_if_sequential 是否使用 seq 特征
# embed_dim 缺省 embed_size
# embed_cate_if_output2d : sparse 特征是否只输出 2 维 （batch, sparse_fea_num * emb_dim）
# inputs_if_embed_dense : 是否对数组特征进行 emb 编码 然后和 sparse 合并：（？, sparse_fea_num + numerical_fea_num, emb_dim）， 什么操作不懂，数值代表权重？
# outputs_dense_if_add_sparse： 是否把 sparse emb 加到数值特征:（？, sparse_fea_num * emb_dim + numerical_fea_num)
#

class InputsLayer(tf.keras.layers.Layer):
    def __init__(self, feature_names, seed=None,
                 inputs_mode='Inputs:feature',
                 inputs_if_multivalued=False, inputs_if_sequential=False,
                 inputs_if_embed_dense=False, outputs_dense_if_add_sparse=True,
                 embed_dim=32, embed_l2=0.0, embed_initializer=None,
                 embed_2d_dim=None, embed_cate_if_output2d=False,
                 pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_l2=0.0, pool_mv_initializer=None,
                 pool_seq_mode='Pooling:average', pool_seq_axis=2, pool_seq_l2=0.0, pool_seq_initializer=None,
                 ):
        super().__init__()
        if inputs_mode not in InputsModeList:
            raise MLGBError

        dense_feature_names, sparse_feature_names, mv_feature_names, seq_feature_names = FeatureNamesLayer(
            inputs_if_multivalued=inputs_if_multivalued, 
            inputs_if_sequential=inputs_if_sequential,
        ).get(feature_names)

        # 对 sparse 特征 emb lookup
        self.embed_cate_fn = SparseEmbeddingLayer(
            sparse_feature_names=sparse_feature_names,
            embed_dim=embed_2d_dim if embed_cate_if_output2d else embed_dim,
            embed_l2=embed_l2,
            embed_initializer=embed_initializer,
            embed_if_output2d=embed_cate_if_output2d,
            seed=seed,
        )

        self.embed_mv_fn = SparseEmbeddingLayer(
            sparse_feature_names=mv_feature_names,
            embed_dim=embed_dim,
            embed_l2=embed_l2,
            embed_initializer=embed_initializer,
            embed_if_output2d=False,
            seed=seed,
        ) if inputs_if_multivalued else None
        self.embed_seq_fn = SparseEmbeddingLayer(
            sparse_feature_names=seq_feature_names,
            embed_dim=embed_dim,
            embed_l2=embed_l2,
            embed_initializer=embed_initializer,
            embed_if_output2d=False,
            seed=seed,
        ) if inputs_if_sequential else None
        # 对数值特征 进行 emb lookup ？
        self.embed_dense_fn = DenseEmbeddingLayer(
            embed_dim=embed_dim,
            embed_l2=embed_l2,
            embed_initializer=embed_initializer,
            seed=seed,
        ) if inputs_if_embed_dense else None
        self.pool_mv_fn = MultiValuedPoolingLayer(
            pool_if_output2d=False,
            pool_mode=pool_mv_mode,
            pool_axis=pool_mv_axis,
            pool_l2=pool_mv_l2,
            pool_initializer=pool_mv_initializer,
            seed=seed,
        ) if inputs_if_multivalued else None
        self.pool_seq_fn = MultiValuedPoolingLayer(
            pool_if_output2d=False,
            pool_mode=pool_seq_mode,
            pool_axis=pool_seq_axis,
            pool_l2=pool_seq_l2,
            pool_initializer=pool_seq_initializer,
            seed=seed,
        ) if inputs_if_sequential else None

        if inputs_mode == 'Inputs:feature':
            print("#InputLayer, input_mode: Inputs:feature")
            # 输出 dense_2d_tensor, embed_cate_3d_tensor
            self.input_fn = FeatureInputsLayer(
                inputs_if_multivalued=inputs_if_multivalued,
                inputs_if_sequential=inputs_if_sequential,
                inputs_if_embed_dense=inputs_if_embed_dense,
                outputs_dense_if_add_sparse=outputs_dense_if_add_sparse,
                embed_sparse_cate_fn=self.embed_cate_fn,
                embed_sparse_mv_fn=self.embed_mv_fn,
                embed_sparse_seq_fn=self.embed_seq_fn,
                embed_dense_fn=self.embed_dense_fn,
                pool_mv_fn=self.pool_mv_fn,
                pool_seq_fn=self.pool_seq_fn,
            )
        else:
            # 输出 dense_2d_tensor, embed_cate_3d_tensor, seq_3d_tensor
            self.input_fn = SequentialInputsLayer(
                inputs_if_multivalued=inputs_if_multivalued,
                inputs_if_sequential=inputs_if_sequential,
                inputs_if_embed_dense=inputs_if_embed_dense,
                outputs_dense_if_add_sparse=outputs_dense_if_add_sparse,
                embed_sparse_cate_fn=self.embed_cate_fn,
                embed_sparse_mv_fn=self.embed_mv_fn,
                embed_sparse_seq_fn=self.embed_seq_fn,
                embed_dense_fn=self.embed_dense_fn,
                pool_mv_fn=self.pool_mv_fn,
                pool_seq_fn=self.pool_seq_fn,
            )

    def build(self, input_shape):
        # [dens_input, sparse_input, (mv_input), (seq_input)] 后俩个可选
        if len(input_shape) not in (2, 3, 4):
            raise MLGBError

        self.built = True
        return

    @tf.function
    def call(self, inputs):
        outputs = self.input_fn(inputs)
        print("#InputsLayer 1, outputs size: ", len(outputs))
        print("#InputsLayer 2, outputs 0 shape: ", outputs[0].shape)
        print("#InputsLayer 3, outputs 1 shape: ", outputs[1].shape)
        return outputs

