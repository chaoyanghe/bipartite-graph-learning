# 系统需求
- `python`
- `sklearn 0.20.0` 以上
 
# 运行前准备
- 运行程序之前，输入命令：
~~~
source ./set_dynamic_lib.sh 	
~~~
进行动态库的链接， 如果使用`run_demo.sh` 则不需要。

- 解压`data.zip` 到根目录。得到`data\` 文件夹。

# 文件描述
工程文件描述
 - `bin\pn2v_opt`: Graph Embedding 实现。 
 -  `bin\pn2v_opt_mpi`: Graph Embedding 的mpi 版本
 - `bin\rd_w`: Biased Random Walk 实现。 
 - `data\`: 数据文件夹，数据格式见后输入格式描述。  
 - `conf.py`: 参数文件，包含运行中的一些可调整参数。
 - `logistic_regression.py`：示例用逻辑回归任务，输入节点属性文件和Graph Embedding生成文件，输出分类结果。注意：这个文件中的`load_node_attr(fname, node_idx_name)`函数需要根据`node_attr`数据不同而作出相应修改。
 - `gen_emb.py`: Embedding生成，输入`edgelist`, `node_list`，生成emb文件。
 - `run_demo.sh`: 运行demo。

# 输入格式描述
 - `node_list`: 节点名同节点id的映射信息。算法输入需要连续整数作为点id，所以需要对原始的名字做一个重新映射。文件格式为：
   ~~~
   节点原始名\t节点id
   ~~~
   示例：
   ~~~
   4499e9e83af903f64c259445258785a9	1
   ~~~
 - `edgelist`: 图的边信息，保存边关系和权重。格式为：
   ~~~
   起始点id\t目标点id\t边权重
   ~~~
   示例：
   ~~~
   0	1	1.2
   ~~~
 - `node_attr`: 点属性信息，用于demo中分类。格式为：
   ~~~
   节点id\t属性1\t属性2\t
   ~~~
   示例：
   ~~~
   2	3032	13	120	186	30	0
   ~~~
 - `node_true`: 真实节点标签列表，用于demo中分类。

# 输出格式描述
 - `tmp.emb`: 算法生成的embedding文件，格式：
   ~~~
   节点id 维度1 维度2 ...
   ~~~
   示例：
   ~~~
   3938 -0.487039 0.252507 -0.463986 0.798046 -0.486098
   ~~~


# 使用方法
执行demo，生成embedding向量并执行逻辑回归。
需要`edgelist`, `node_attr`, `node_list`, `node_true` 四个文件(后续会提供)。
~~~
sh run_demo.sh
~~~

生成embedding向量，仅生成embedding向量。
需要`edgelist`, `node_list`两个文件(后续会提供)。会在out文件夹输出tmp.emb文件。
~~~
python gen_emb.py
~~~
