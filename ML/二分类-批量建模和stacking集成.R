# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类

##############################################################

# install.packages("tidymodels")
library(tidymodels)
library(bonsai)
source("tidyfuncs4cls2.R")

# 读取数据
# read.csv()
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

# 变量类型修正后数据概况
skimr::skim(Heart)

# 设定阳性类别和阴性类别
yourpositivelevel <- "Yes"
yournegativelevel <- "No"

# 转换因变量的因子水平，将阳性类别设定为第二个水平
levels(Heart$AHD)
table(Heart$AHD)
Heart$AHD <- factor(
  Heart$AHD,
  levels = c(yournegativelevel, yourpositivelevel)
)
levels(Heart$AHD)
table(Heart$AHD)

# 数据拆分
set.seed(42)
datasplit <- initial_split(Heart, prop = 0.75, strata = AHD)
traindata <- training(datasplit)
testdata <- testing(datasplit)

# 重抽样设定-5折交叉验证
set.seed(42)
folds <- vfold_cv(traindata, v = 5, strata = AHD)
folds

##############################################################

# 批量建模之前做好各个模型的设定
# 模型对应的预处理
# 模型
# 模型的超参数调优网格

####################### 决策树模型

# 数据预处理配方
datarecipe_dt <- recipe(AHD ~ ., traindata) %>%
  prep()
datarecipe_dt

# 设定模型
model_dt <- decision_tree(
  mode = "classification",
  engine = "rpart",
  tree_depth = tune(),
  min_n = tune(),
  cost_complexity = tune()
) %>%
  set_args(model=TRUE)
model_dt

# 超参数寻优网格
set.seed(42)
hpgrid_dt <- parameters(
  tree_depth(range = c(3, 7)),
  min_n(range = c(5, 10)),
  cost_complexity(range = c(-6, -1))
) %>%
  grid_random(size = 10) # 随机网格
hpgrid_dt

####################### 随机森林模型

# 数据预处理配方
datarecipe_rf <- recipe(AHD ~ ., traindata) %>%
  prep()
datarecipe_rf

# 设定模型
model_rf <- rand_forest(
  mode = "classification",
  engine = "randomForest", # ranger
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_args(importance = T)
model_rf

# 超参数寻优网格
set.seed(42)
hpgrid_rf <- parameters(
  mtry(range = c(2, 10)), 
  trees(range = c(200, 500)),
  min_n(range = c(20, 50))
) %>%
  grid_random(size = 5) # 随机网格
hpgrid_rf

############################ Xgboost

# 数据预处理配方
datarecipe_xgboost <- recipe(AHD ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), 
             naming = new_dummy_names) %>% 
  prep()
datarecipe_xgboost

# 设定模型
model_xgboost <- boost_tree(
  mode = "classification",
  engine = "xgboost",
  mtry = tune(),
  trees = 1000,
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  stop_iter = 25
) %>%
  set_args(validation = 0.2,
           event_level = "second")
model_xgboost

# 超参数寻优网格
set.seed(42)
hpgrid_xgboost <- parameters(
  mtry(range = c(2, 8)),
  min_n(range = c(5, 20)),
  tree_depth(range = c(1, 3)),
  learn_rate(range = c(-3, -1)),
  loss_reduction(range = c(-3, 0)),
  sample_prop(range = c(0.8, 1))
) %>%
  grid_random(size = 5) # 随机网格
hpgrid_xgboost


############################################################

################################ 批量寻优2选1-网格搜索

wk_set <- workflow_set(
  preproc = list(dr4dt = datarecipe_dt,
                 dr4rf = datarecipe_rf,
                 dr4xgboost = datarecipe_xgboost), 
  models = list(dt = model_dt, 
                rf = model_rf,
                xgboost = model_xgboost),
  cross = F
)
wk_set

wk_set <- wk_set %>%
  option_add(grid = hpgrid_dt, id = "dr4dt_dt") %>%
  option_add(grid = hpgrid_rf, id = "dr4rf_rf") %>%
  option_add(grid = hpgrid_xgboost, id = "dr4xgboost_xgboost")
wk_set

tune_set <- wk_set %>% 
  workflow_map(
    fn = "tune_grid",
    verbose = TRUE,
    seed = 42,
    resamples = folds,
    metrics = metric_set(yardstick::roc_auc),
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )
tune_set

################################ 批量寻优2选1-贝叶斯优化

wk_set <- workflow_set(
  preproc = list(dr4dt = datarecipe_dt,
                 dr4rf = datarecipe_rf,
                 dr4xgboost = datarecipe_xgboost), 
  models = list(dt = model_dt, 
                rf = model_rf,
                xgboost = model_xgboost),
  cross = F
)
wk_set

param_rf <- model_rf %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(2, 10)))
param_xgboost <- model_xgboost %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(2, 10)))

wk_set <- wk_set %>%
  option_add(param_info = param_rf, id = "dr4rf_rf") %>%
  option_add(param_info = param_xgboost, id = "dr4xgboost_xgboost")
wk_set

tune_set <- wk_set %>% 
  workflow_map(
    fn = "tune_bayes",
    verbose = TRUE,
    seed = 42,
    resamples = folds,
    initial = 10,
    iter = 50,
    metrics = metric_set(yardstick::roc_auc),
    control = control_bayes(save_pred = T, 
                           verbose = T,
                           no_improve = 10,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )
tune_set

################################
################################

# 各模型最优超参数的预测结果
tune_prediction <- tune_set %>%
  collect_predictions(select_best = T,)
colnames(tune_prediction)

# 各模型最优超参数交叉验证评估结果
tune_metrics <- tune_set %>%
  collect_metrics() %>%
  inner_join(tune_prediction %>%
               distinct(wflow_id, .config))

# 图示
tune_metrics %>%
  ggplot(aes(x = wflow_id, y = mean)) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = mean-std_err, 
                    ymax = mean+std_err),
                width = 0.1, linewidth = 1) +
  # scale_y_continuous(limits = c(0, 1)) +
  labs(y = "cv roc_auc", x = "") +
  theme_bw()

################################

# 提取最优模型的最优参数
hpbest <- tune_set %>%
  extract_workflow_set_result("dr4rf_rf") %>%
  select_best(metric = "roc_auc")
hpbest

# 最优模型
final_model <-  tune_set %>% 
  extract_workflow("dr4rf_rf") %>% 
  finalize_workflow(hpbest) %>% 
  fit(traindata)
final_model

################################################################

# stacking模型中可能包括不同类型的模型，也可能包括相同类型的模型的不同配置

# 用于构建stacking模型的样本自变量值是候选基础模型的样本外预测结果

library(stacks)

##############################
# 基础模型设定
# 可以是之前批量建模的结果
models_stack <- 
  stacks() %>% 
  add_candidates(tune_set)
models_stack
# 也可以是之前单个模型建模的结果
load(".\\cls2\\evalresult_knn.RData")
load(".\\cls2\\evalresult_rf.RData")
load(".\\cls2\\evalresult_logistic.RData")
models_stack <- 
  stacks() %>% 
  add_candidates(tune_knn) %>%
  add_candidates(tune_rf) %>%
  add_candidates(cv_logistic)
models_stack

##############################

# 拟合stacking元模型——lasso
set.seed(42)
meta_stack <- blend_predictions(
  models_stack, 
  penalty = 10^seq(-2, -0.5, length = 20)
)
meta_stack
autoplot(meta_stack)

# 拟合选定的基础模型
set.seed(42)
fit_stack <- fit_members(meta_stack)

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_stack_heart <- fit_stack
traindata_heart <- traindata
save(final_stack_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_stack_heart.RData")

######################################################

# 以下遭遇 lightgbm 会报错
# 应用stacking模型预测并评估

# 训练集
predtrain_stack <- eval4cls2(
  model = fit_stack, 
  dataset = traindata, 
  yname = "AHD", 
  modelname = "stacking", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_stack$prediction
predtrain_stack$predprobplot
predtrain_stack$rocresult
predtrain_stack$rocplot
predtrain_stack$prresult
predtrain_stack$prplot
predtrain_stack$cmresult
predtrain_stack$cmplot
predtrain_stack$metrics
predtrain_stack$diycutoff

# pROC包auc值及其置信区间
pROC::auc(predtrain_stack$proc)
pROC::ci.auc(predtrain_stack$proc)

# 测试集
predtest_stack <- eval4cls2(
  model = fit_stack, 
  dataset = testdata, 
  yname = "AHD", 
  modelname = "stacking", 
  datasetname = "testdata",
  cutoff = predtrain_stack$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_stack$prediction
predtest_stack$predprobplot
predtest_stack$rocresult
predtest_stack$rocplot
predtest_stack$prresult
predtest_stack$prplot
predtest_stack$cmresult
predtest_stack$cmplot
predtest_stack$metrics
predtest_stack$diycutoff

# pROC包auc值及其置信区间
pROC::auc(predtest_stack$proc)
pROC::ci.auc(predtest_stack$proc)

# ROC比较检验
pROC::roc.test(predtrain_stack$proc, predtest_stack$proc)

###################################################################
######################## DALEX解释对象
# 自变量数据集
colnames(traindata)
traindatax <- traindata %>%
  dplyr::select(-AHD)
colnames(traindatax)

explainer_stack <- DALEXtra::explain_tidymodels(
  fit_stack, 
  data = traindatax,
  y = ifelse(traindata$AHD == yourpositivelevel, 1, 0),
  type = "classification",
  label = "stacking"
)
# 变量重要性
set.seed(42)
vip_stack <- DALEX::model_parts(
  explainer_stack,
  type = "ratio"
)
plot(vip_stack)

# 变量偏依赖图
# 连续型变量
set.seed(42)
pdpc_stack <- DALEX::model_profile(
  explainer_stack,
  variables = colnames(traindatax)
)
plot(pdpc_stack)
# 分类变量
set.seed(42)
pdpd_stack <- DALEX::model_profile(
  explainer_stack,
  variables = colnames(traindatax)[c(2,3,6,7,9,11,13)]  # 分类变量所在位置
)
plot(pdpd_stack)

# 单样本预测分解
set.seed(42)
shap_stack <- DALEX::predict_parts(
  explainer = explainer_stack, 
  new_observation = traindatax[2, ], 
  type = "shap"
)
plot(shap_stack)


######################## fastshap包

shapresult <- shap4cls2(
  finalmodel = fit_stack,
  predfunc = function(model, newdata) {
    predict(model, newdata, type = "prob") %>%
      select(ends_with(yourpositivelevel)) %>%
      pull()
  },
  datax = traindatax,
  datay = traindata$AHD,
  yname = "AHD",
  flname = colnames(traindatax)[c(2,3,6,7,9,11,13)],
  lxname = colnames(traindatax)[-c(2,3,6,7,9,11,13)]
)

# 基于shap的变量重要性
shapresult$shapvip
# 单样本预测分解
fastshap::force_plot(
  object = shapresult$shapley[1, ], 
  feature_values = as.data.frame(traindatax)[1, ], 
  baseline = mean(predtrain_stack$prediction$.pred_Yes),
  display = "viewer"
)
# 所有分类变量的shap图示
shapresult$shapplotd_facet
shapresult$shapplotd_one
# 所有连续变量的shap图示
shapresult$shapplotc_facet
shapresult$shapplotc_one
# 单分类变量shap图示
shap1d <- shapresult$shapdatad %>%
  dplyr::filter(feature == "ChestPain") %>% # 某个要展示的分类自变量
  na.omit() %>%
  ggplot(aes(x = value, y = shap)) +
  geom_boxplot(fill = "lightgreen") +
  geom_point(aes(color = Y), alpha = 0.5) + 
  geom_hline(yintercept = 0, color = "grey10") +
  scale_color_viridis_d() +
  labs(x = "ChestPain", color = "AHD") + # 自变量名称和因变量名称
  theme_bw()
shap1d
ggExtra::ggMarginal(
  shap1d + theme(legend.position = "bottom"),
  type = "histogram",
  margins = "y",
  fill = "skyblue"
)

# 单连续变量shap图示
shap1c <- shapresult$shapdatac %>%
  dplyr::filter(feature == "Age") %>% # 某个要展示的连续自变量
  na.omit() %>%
  ggplot(aes(x = value, y = shap)) +
  geom_point(aes(color = Y)) +
  geom_smooth(color = "red") +
  geom_hline(yintercept = 0, color = "grey10") +
  scale_color_viridis_d() +
  labs(x = "Age", color = "AHD") + # 自变量名称和因变量名称
  theme_bw()
shap1c
ggExtra::ggMarginal(
  shap1c + theme(legend.position = "bottom"),
  type = "histogram",
  fill = "skyblue"
)









