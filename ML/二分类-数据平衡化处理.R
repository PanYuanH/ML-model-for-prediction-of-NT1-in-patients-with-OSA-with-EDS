# 模型机器---R语言tidymodels包机器学习分类与回归模型---分类数据平衡化处理

##############################################################

library(tidymodels)
library(themis)

##############################################################

# 二分类数据

# 读取数据
Heart <- readr::read_csv(file.choose())
colnames(Heart) 
# 修正变量类型
# 将分类变量转换为factor
for(i in c(3,4,7,8,10,12,14,15)){ 
  Heart[[i]] <- factor(Heart[[i]])
}

# 删除无关变量在此处进行
Heart$Id <- NULL
# 删除含有缺失值的样本在此处进行，填充缺失值在后面
Heart <- na.omit(Heart)

# 数据拆分
set.seed(4321)
datasplit <- initial_split(Heart, prop = 0.75, strata = AHD)
traindata <- training(datasplit)
testdata <- testing(datasplit)


# 数据预处理
# 先对照训练集写配方
datarecipe <- recipe(AHD ~ ., traindata) %>%
  # 以下选一即可
  # # 1-欠采样
  # step_downsample(AHD, under_ratio = 1) %>%
  # # 2-过采样
  # step_upsample(AHD, over_ratio = 1) %>%
  # # 3-smote，数值型输入
  # step_dummy(all_nominal_predictors()) %>%
  # step_smote(AHD, over_ratio = 1) %>%
  # # 4-rose
  # step_rose(AHD, over_ratio = 1) %>%
  # 5-ADASYN
  step_dummy(all_nominal_predictors()) %>%
  step_adasyn(AHD, over_ratio = 1) %>%
  prep()
datarecipe

# 按方处理训练集和测试集
traindata2 <- bake(datarecipe, new_data = NULL) %>%
  select(AHD, everything())
testdata2 <- bake(datarecipe, new_data = testdata) %>%
  select(AHD, everything())

# 因变量取值频数对比
table(Heart$AHD)
table(traindata$AHD)
table(testdata$AHD)
table(traindata2$AHD)
table(testdata2$AHD)

