вр
к¤
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8░У
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
В
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
В
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
с
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ь
valueТBП BИ
Ш
layer-0
layer_with_weights-0
	optimizer
	keras_api

signatures
regularization_losses
	variables
trainable_variables
h

kernel
	bias

	keras_api
regularization_losses
	variables
trainable_variables
d
iter

beta_1

beta_2
	decay
learning_ratem"	m#v$	v%
н
layer_metrics
trainable_variables

layers
metrics
layer_regularization_losses
non_trainable_variables
regularization_losses
	variables
 
 

0
	1

0
	1
KI
VARIABLE_VALUEdense/kernel)layer-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUE
dense/bias'layer-0/bias/.ATTRIBUTES/VARIABLE_VALUE
н
layer_metrics
trainable_variables
metrics

layers
layer_regularization_losses
non_trainable_variables
regularization_losses
	variables
 

0
	1

0
	1
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
 
 
 
 
 
 
 
4
	total
	count
 	keras_api
!	variables
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

!	variables

0
1
nl
VARIABLE_VALUEAdam/dense/kernel/mElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/dense/bias/mClayer-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense/kernel/vElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/dense/bias/vClayer-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
serving_default_dense_inputPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
╗
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_inputdense/kernel
dense/bias*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*5
config_proto%#

CPU

GPU2*0J

  АD8*+
f&R$
"__inference_signature_wrapper_9601
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╨
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J

  АD8*&
f!R
__inference__traced_save_9724
╟
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense/kernel/mAdam/dense/bias/mAdam/dense/kernel/vAdam/dense/bias/v*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J

  АD8*)
f$R"
 __inference__traced_restore_9775╨▐
┘
и
D__inference_sequential_layer_call_and_return_conditional_losses_9545
dense_input

dense_9539

dense_9541
identityИвdense/StatefulPartitionedCallэ
dense/StatefulPartitionedCallStatefulPartitionedCalldense_input
dense_9539
dense_9541*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*5
config_proto%#

CPU

GPU2*0J

  АD8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_95192
dense/StatefulPartitionedCallЪ
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:T P
'
_output_shapes
:         
%
_user_specified_namedense_input:

_output_shapes
: :

_output_shapes
: 
ъ
╕
D__inference_sequential_layer_call_and_return_conditional_losses_9611

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИЯ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOpЕ
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/BiasAddj
IdentityIdentitydense/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
С
Г
)__inference_sequential_layer_call_fn_9582
dense_input
unknown
	unknown_0
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*5
config_proto%#

CPU

GPU2*0J

  АD8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_95752
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:         
%
_user_specified_namedense_input:

_output_shapes
: :

_output_shapes
: 
┘
и
D__inference_sequential_layer_call_and_return_conditional_losses_9536
dense_input

dense_9530

dense_9532
identityИвdense/StatefulPartitionedCallэ
dense/StatefulPartitionedCallStatefulPartitionedCalldense_input
dense_9530
dense_9532*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*5
config_proto%#

CPU

GPU2*0J

  АD8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_95192
dense/StatefulPartitionedCallЪ
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:T P
'
_output_shapes
:         
%
_user_specified_namedense_input:

_output_shapes
: :

_output_shapes
: 
Б
~
)__inference_sequential_layer_call_fn_9630

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*5
config_proto%#

CPU

GPU2*0J

  АD8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_95572
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╩
г
D__inference_sequential_layer_call_and_return_conditional_losses_9557

inputs

dense_9551

dense_9553
identityИвdense/StatefulPartitionedCallш
dense/StatefulPartitionedCallStatefulPartitionedCallinputs
dense_9551
dense_9553*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*5
config_proto%#

CPU

GPU2*0J

  АD8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_95192
dense/StatefulPartitionedCallЪ
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╩
г
D__inference_sequential_layer_call_and_return_conditional_losses_9575

inputs

dense_9569

dense_9571
identityИвdense/StatefulPartitionedCallш
dense/StatefulPartitionedCallStatefulPartitionedCallinputs
dense_9569
dense_9571*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*5
config_proto%#

CPU

GPU2*0J

  АD8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_95192
dense/StatefulPartitionedCallЪ
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Д
з
?__inference_dense_layer_call_and_return_conditional_losses_9649

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Д
з
?__inference_dense_layer_call_and_return_conditional_losses_9519

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╞,
╨
__inference__traced_save_9724
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1П
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_e17062202e4d42b2ab8664a186ae3385/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename└
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╥
value╚B┼B)layer-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-0/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesв
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices╘
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardм
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1в
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices╧
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesм
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*U
_input_shapesD
B: ::: : : : : : : ::::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :$
 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
Б
~
)__inference_sequential_layer_call_fn_9639

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*5
config_proto%#

CPU

GPU2*0J

  АD8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_95752
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ф
|
"__inference_signature_wrapper_9601
dense_input
unknown
	unknown_0
identityИвStatefulPartitionedCall╜
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*5
config_proto%#

CPU

GPU2*0J

  АD8*(
f#R!
__inference__wrapped_model_95052
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:         
%
_user_specified_namedense_input:

_output_shapes
: :

_output_shapes
: 
ў
y
$__inference_dense_layer_call_fn_9658

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*5
config_proto%#

CPU

GPU2*0J

  АD8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_95192
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Д

о
__inference__wrapped_model_9505
dense_input3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identityИ└
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&sequential/dense/MatMul/ReadVariableOpл
sequential/dense/MatMulMatMuldense_input.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/dense/MatMul┐
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp┼
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/dense/BiasAddu
IdentityIdentity!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::T P
'
_output_shapes
:         
%
_user_specified_namedense_input:

_output_shapes
: :

_output_shapes
: 
ъ
╕
D__inference_sequential_layer_call_and_return_conditional_losses_9621

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИЯ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOpЕ
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/BiasAddj
IdentityIdentitydense/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
С
Г
)__inference_sequential_layer_call_fn_9564
dense_input
unknown
	unknown_0
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*5
config_proto%#

CPU

GPU2*0J

  АD8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_95572
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:         
%
_user_specified_namedense_input:

_output_shapes
: :

_output_shapes
: 
▓=
▄
 __inference__traced_restore_9775
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate
assignvariableop_7_total
assignvariableop_8_count*
&assignvariableop_9_adam_dense_kernel_m)
%assignvariableop_10_adam_dense_bias_m+
'assignvariableop_11_adam_dense_kernel_v)
%assignvariableop_12_adam_dense_bias_v
identity_14ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9в	RestoreV2вRestoreV2_1╞
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╥
value╚B┼B)layer-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-0/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesи
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesь
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityН
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1У
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0	*
_output_shapes
:2

Identity_2Т
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Ф
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Ф
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5У
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Ы
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7О
AssignVariableOp_7AssignVariableOpassignvariableop_7_totalIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8О
AssignVariableOp_8AssignVariableOpassignvariableop_8_countIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9Ь
AssignVariableOp_9AssignVariableOp&assignvariableop_9_adam_dense_kernel_mIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10Ю
AssignVariableOp_10AssignVariableOp%assignvariableop_10_adam_dense_bias_mIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11а
AssignVariableOp_11AssignVariableOp'assignvariableop_11_adam_dense_kernel_vIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12Ю
AssignVariableOp_12AssignVariableOp%assignvariableop_12_adam_dense_bias_vIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12и
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp№
Identity_13Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_13Й
Identity_14IdentityIdentity_13:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_14"#
identity_14Identity_14:output:0*I
_input_shapes8
6: :::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "пL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*░
serving_defaultЬ
C
dense_input4
serving_default_dense_input:0         9
dense0
StatefulPartitionedCall:0         tensorflow/serving/predict:▄E
▓
layer-0
layer_with_weights-0
	optimizer
	keras_api

signatures
regularization_losses
	variables
trainable_variables
&_default_save_signature
*'&call_and_return_all_conditional_losses
(__call__"└
_tf_keras_sequentialб{"class_name": "Sequential", "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "training_config": {"weighted_metrics": null, "loss": "mean_squared_error", "loss_weights": null, "metrics": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"amsgrad": false, "name": "Adam", "decay": 0.0, "beta_1": 0.8999999761581421, "learning_rate": 0.10000000149011612, "epsilon": 1e-07, "beta_2": 0.9990000128746033}}}, "model_config": {"class_name": "Sequential", "config": {"build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "name": "sequential", "layers": [{"class_name": "Dense", "config": {"name": "dense", "bias_regularizer": null, "dtype": "float32", "activation": "linear", "bias_constraint": null, "trainable": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "activity_regularizer": null, "units": 1, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "use_bias": true, "kernel_regularizer": null}}]}}, "keras_version": "2.3.0-tf", "name": "sequential", "is_graph_network": true, "dtype": "float32", "trainable": true, "config": {"build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "name": "sequential", "layers": [{"class_name": "Dense", "config": {"name": "dense", "bias_regularizer": null, "dtype": "float32", "activation": "linear", "bias_constraint": null, "trainable": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "activity_regularizer": null, "units": 1, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "use_bias": true, "kernel_regularizer": null}}]}, "backend": "tensorflow", "expects_training_arg": true, "batch_input_shape": null, "input_spec": {"class_name": "InputSpec", "config": {"axes": {"-1": 1}, "ndim": null, "shape": null, "dtype": null, "min_ndim": 2, "max_ndim": null}}}
╣

kernel
	bias

	keras_api
regularization_losses
	variables
trainable_variables
*)&call_and_return_all_conditional_losses
*__call__"Ф
_tf_keras_layer·{"expects_training_arg": false, "trainable": true, "stateful": false, "config": {"name": "dense", "bias_regularizer": null, "dtype": "float32", "bias_constraint": null, "activation": "linear", "trainable": true, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "activity_regularizer": null, "units": 1, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "use_bias": true, "kernel_regularizer": null}, "name": "dense", "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "class_name": "Dense", "input_spec": {"class_name": "InputSpec", "config": {"axes": {"-1": 1}, "ndim": null, "shape": null, "dtype": null, "min_ndim": 2, "max_ndim": null}}}
w
iter

beta_1

beta_2
	decay
learning_ratem"	m#v$	v%"
	optimizer
╩
layer_metrics
trainable_variables

layers
metrics
layer_regularization_losses
non_trainable_variables
regularization_losses
	variables
&_default_save_signature
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses
(__call__"
_generic_user_object
,
+serving_default"
signature_map
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
:2dense/kernel
:2
dense/bias
н
layer_metrics
trainable_variables
metrics

layers
layer_regularization_losses
non_trainable_variables
regularization_losses
	variables
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses
*__call__"
_generic_user_object
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╗
	total
	count
 	keras_api
!	variables"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "config": {"name": "loss", "dtype": "float32"}, "dtype": "float32"}
:  (2total
:  (2count
-
!	variables"
_generic_user_object
.
0
1"
trackable_list_wrapper
#:!2Adam/dense/kernel/m
:2Adam/dense/bias/m
#:!2Adam/dense/kernel/v
:2Adam/dense/bias/v
с2▐
__inference__wrapped_model_9505║
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк **в'
%К"
dense_input         
▐2█
D__inference_sequential_layer_call_and_return_conditional_losses_9621
D__inference_sequential_layer_call_and_return_conditional_losses_9611
D__inference_sequential_layer_call_and_return_conditional_losses_9536
D__inference_sequential_layer_call_and_return_conditional_losses_9545└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Є2я
)__inference_sequential_layer_call_fn_9639
)__inference_sequential_layer_call_fn_9564
)__inference_sequential_layer_call_fn_9630
)__inference_sequential_layer_call_fn_9582└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
щ2ц
?__inference_dense_layer_call_and_return_conditional_losses_9649в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╬2╦
$__inference_dense_layer_call_fn_9658в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
5B3
"__inference_signature_wrapper_9601dense_inputМ
__inference__wrapped_model_9505i	4в1
*в'
%К"
dense_input         
к "-к*
(
denseК
dense         Я
?__inference_dense_layer_call_and_return_conditional_losses_9649\	/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ w
$__inference_dense_layer_call_fn_9658O	/в,
%в"
 К
inputs         
к "К         ▒
D__inference_sequential_layer_call_and_return_conditional_losses_9536i	<в9
2в/
%К"
dense_input         
p

 
к "%в"
К
0         
Ъ ▒
D__inference_sequential_layer_call_and_return_conditional_losses_9545i	<в9
2в/
%К"
dense_input         
p 

 
к "%в"
К
0         
Ъ м
D__inference_sequential_layer_call_and_return_conditional_losses_9611d	7в4
-в*
 К
inputs         
p

 
к "%в"
К
0         
Ъ м
D__inference_sequential_layer_call_and_return_conditional_losses_9621d	7в4
-в*
 К
inputs         
p 

 
к "%в"
К
0         
Ъ Й
)__inference_sequential_layer_call_fn_9564\	<в9
2в/
%К"
dense_input         
p

 
к "К         Й
)__inference_sequential_layer_call_fn_9582\	<в9
2в/
%К"
dense_input         
p 

 
к "К         Д
)__inference_sequential_layer_call_fn_9630W	7в4
-в*
 К
inputs         
p

 
к "К         Д
)__inference_sequential_layer_call_fn_9639W	7в4
-в*
 К
inputs         
p 

 
к "К         Ю
"__inference_signature_wrapper_9601x	Cв@
в 
9к6
4
dense_input%К"
dense_input         "-к*
(
denseК
dense         