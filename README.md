# 使用 AWS SageMaker 部署机器学习模型

代码和相关文件 

此代码库包含使用 AWS SageMaker 部署机器学习模型所需的代码和相关文件，并且由各种编程练习的多个教程 notebook、迷你项目和实战项目文件组成，是对这门纳米学位课程的补充。

## 目录

### 教程
* [Boston Housing (Batch Transform) - High Level](https://github.com/udacity/sagemaker-deployment/tree/master/Tutorials/Boston%20Housing%20-%20XGBoost%20(Batch%20Transform)%20-%20High%20Level.ipynb) 是最简单的 notebook，介绍了 SageMaker 生态系统以及所有组件是如何在一起运行的。使用的数据已经清理过并且是表格形式，不需要再处理数据了。请使用批量转换方法测试拟合的模型。
* [Boston Housing (Batch Transform) - Low Level](https://github.com/udacity/sagemaker-deployment/tree/master/Tutorials/Boston%20Housing%20-%20XGBoost%20(Batch%20Transform)%20-%20Low%20Level.ipynb) 和高阶 notebook 执行的分析一样，但是使用的是低阶 API。所以更详细，但是更加灵活。建议了解每种方法，即使仅使用其中某个方法。
* [Boston Housing (Deploy) - High Level](https://github.com/udacity/sagemaker-deployment/blob/master/Tutorials/Boston%20Housing%20-%20XGBoost%20(Deploy)%20-%20High%20Level.ipynb) 是同一名称的 Batch Transform notebook 的变体。它没有使用批量转换测试模型，而是部署模型并将测试数据发送给部署的端点。
* [Boston Housing (Deploy) - Low Level](https://github.com/udacity/sagemaker-deployment/blob/master/Tutorials/Boston%20Housing%20-%20XGBoost%20(Deploy)%20-%20Low%20Level.ipynb) 也是上述 Batch Transform notebook 的变体。这次使用低阶 API，并且部署模型并发送数据，而不是使用批量转换方法。
* [IMDB Sentiment Analysis - XGBoost - Web App](https://github.com/udacity/sagemaker-deployment/blob/master/Tutorials/IMDB%20Sentiment%20Analysis%20-%20XGBoost%20-%20Web%20App.ipynb) 使用 XGBoost 创建一个情感分析模型，并将模型部署到端点上。然后描述如何设置 AWS Lambda 和 API Gateway 以创建一个简单的网络应用，该应用会与部署的端点交互。
* [Boston Housing (Hyperparameter Tuning) - High Level](https://github.com/udacity/sagemaker-deployment/tree/master/Tutorials/Boston%20Housing%20-%20XGBoost%20(Hyperparameter%20Tuning)%20-%20High%20Level.ipynb) 是 Boston Housing XGBoost 模型的扩展，这次并不是训练一个模型，而是使用 SageMaker 的超参数优化功能训练多个不同的模型，最终使用性能最佳的模型。
* [Boston Housing (Hyperparameter Tuning) - Low Level](https://github.com/udacity/sagemaker-deployment/tree/master/Tutorials/Boston%20Housing%20-%20XGBoost%20(Hyperparameter%20Tuning)%20-%20Low%20Level.ipynb) 是高阶超参数优化 notebook 的变体，这次使用低阶 API 创建在构建超参数优化作业时涉及的每个对象。
* [Boston Housing - Updating an Endpoint](https://github.com/udacity/sagemaker-deployment/tree/master/Tutorials/Boston%20Housing%20-%20Updating%20an%20Endpoint.ipynb) 是 Boston Housing XGBoost 模型的另一个扩展，我们构建了一个线性模型并在构建的两个模型之间切换部署的端点。此外，我们将创建一个端点，它会模拟 A/B 测试，将传入的部分推理请求发送给 XGBoost 模型，并将剩下的请求发送给线性模型。

### 迷你项目
* [IMDB Sentiment Analysis - XGBoost (Batch Transform)](https://github.com/udacity/sagemaker-deployment/tree/master/Mini-Projects/IMDB%20Sentiment%20Analysis%20-%20XGBoost%20(Batch%20Transform).ipynb) notebook 需要你来完成，它会指导你使用 XGBoost 构建一个对 IMDB 数据集进行情感分析的模型。
* [IMDB Sentiment Analysis - XGBoost (Hyperparameter Tuning)](https://github.com/udacity/sagemaker-deployment/tree/master/Mini-Projects/IMDB%20Sentiment%20Analysis%20-%20XGBoost%20(Hyperparameter%20Tuning).ipynb) notebook 需要你来完成，它会指导你使用 XGBoost 构建一个情感分析模型，并使用 SageMaker 的超参数优化功能测试多组不同的超参数。
* [IMDB Sentiment Analysis - XGBoost (Updating a Model)](https://github.com/udacity/sagemaker-deployment/tree/master/Mini-Projects/IMDB%20Sentiment%20Analysis%20-%20XGBoost%20(Updating%20a%20Model).ipynb) notebook 需要你来完成，它会指导你使用 XGBoost 构建一个情感分析模型，然后看看当底层数据分布发生变化时，会发生什么。在探索了数据随着时间的推移发生变化后，你将构建一个更新的模型，并更新部署的端点，使其使用新的模型。

### 实战项目

[Sentiment Analysis Web App](https://github.com/udacity/sagemaker-deployment/tree/master/Project) 是一个 notebook 和 Python 文件集合，需要你来完成。结果是一个对影评进行情感分析的部署 RNN，并且需要创建可公共访问的 API 以及与部署的端点交互的简单网络应用。此项目假设你熟悉 SageMaker。完成 XGBoost Sentiment Analysis notebook 就可以了。

## 设置说明

此代码库中提供的 notebook 需要使用 Amazon SageMaker 平台执行。下面简要说明了如何使用 SageMaker 设置托管 notebook 实例，你可以在此实例中完成和运行 notebook。

### 登录 AWS 控制台并创建一个 notebook 实例

登录 AWS 控制台并转到 SageMaker 信息中心。点击“Create notebook instance”。notebook 可以随意命名，建议使用 ml.t2.medium，因为它属于免费套餐。对于角色，新建一个角色就行了。使用默认选项即可。注意，notebook 实例需要能够访问 S3 资源，默认就能访问。该 notebook 可以访问名称中带 sagemaker 的任何 S3 存储桶或对象。

### 使用 git 将代码库克隆到 notebook 实例中

实例启动并能访问后，点击“open”以转到 Jupyter notebook 主页面。首先将 SageMaker Deployment github 代码库克隆到 notebook 实例中。注意，需要克隆到相应的目录中，以便数据能在会话之间保留。

点击“new”下拉菜单并选择“terminal”。默认情况下，终端实例的工作目录是主目录。但是，Jupyter notebook hub 的根目录在“SageMaker”下。请转到相应的目录并克隆代码库，如下所示。

```bash
cd SageMaker
git clone https://github.com/udacity/sagemaker-deployment.git
exit
```

操作完毕后，关闭终端窗口。

### 打开和运行你所选的 notebook

将代码库克隆到 notebook 实例中后，你可以转到要完成或执行的任何 notebook，然后完成该 notebook。每个 notebook 都包含了额外的说明。
