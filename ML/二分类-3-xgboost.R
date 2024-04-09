# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类---xgboost

# https://www.tidymodels.org/find/parsnip/
# https://parsnip.tidymodels.org/reference/boost_tree.html
# https://parsnip.tidymodels.org/reference/details_boost_tree_xgboost.html

# 模型评估指标
# https://cran.r-project.org/web/packages/yardstick/vignettes/metric-types.html

##############################################################

# install.packages("tidymodels")
library(tidymodels)
source("tidyfuncs4cls2.R")

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

##############################################################

# 数据拆分
set.seed(42)
datasplit <- initial_split(Heart, prop = 0.75, strata = AHD)
traindata <- training(datasplit)
testdata <- testing(datasplit)

# 重抽样设定-5折交叉验证
set.seed(42)
folds <- vfold_cv(traindata, v = 5, strata = AHD)
folds

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

# workflow
wk_xgboost <- 
  workflow() %>%
  add_recipe(datarecipe_xgboost) %>%
  add_model(model_xgboost)
wk_xgboost

##############################################################

############################  超参数寻优1-网格搜索

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
  # grid_regular(levels = c(3, 2, 2, 3, 2, 2)) # 常规网格
  grid_random(size = 5) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_xgboost
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_xgboost <- wk_xgboost %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_xgboost,
    metrics = metric_set(yardstick::accuracy, 
                         yardstick::roc_auc, 
                         yardstick::pr_auc),
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )

#########################  超参数寻优2-贝叶斯优化

# 更新超参数范围
param_xgboost <- model_xgboost %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(2, 10)))

# 贝叶斯优化超参数
set.seed(42)
tune_xgboost <- wk_xgboost %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    param_info = param_xgboost,
    metrics = metric_set(yardstick::roc_auc,
                         yardstick::accuracy, 
                         yardstick::pr_auc),
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束


# 图示交叉验证结果
autoplot(tune_xgboost)
eval_tune_xgboost <- tune_xgboost %>%
  collect_metrics()
eval_tune_xgboost

# 经过交叉验证得到的最优超参数
hpbest_xgboost <- tune_xgboost %>%
  select_by_one_std_err(metric = "roc_auc", desc(min_n))
hpbest_xgboost

# 采用最优超参数组合训练最终模型
final_xgboost <- wk_xgboost %>%
  finalize_workflow(hpbest_xgboost) %>%
  fit(traindata)
final_xgboost

##################################################################

# 训练集预测评估
predtrain_xgboost <- eval4cls2(
  model = final_xgboost, 
  dataset = traindata, 
  yname = "AHD", 
  modelname = "Xgboost", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_xgboost$prediction
predtrain_xgboost$predprobplot
predtrain_xgboost$rocresult
predtrain_xgboost$rocplot
predtrain_xgboost$prresult
predtrain_xgboost$prplot
predtrain_xgboost$cmresult
predtrain_xgboost$cmplot
predtrain_xgboost$metrics
predtrain_xgboost$diycutoff

# pROC包auc值及其置信区间
pROC::auc(predtrain_xgboost$proc)
pROC::ci.auc(predtrain_xgboost$proc)

# 预测评估测试集预测评估
predtest_xgboost <- eval4cls2(
  model = final_xgboost, 
  dataset = testdata, 
  yname = "AHD", 
  modelname = "Xgboost", 
  datasetname = "testdata",
  cutoff = predtrain_xgboost$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_xgboost$prediction
predtest_xgboost$predprobplot
predtest_xgboost$rocresult
predtest_xgboost$rocplot
predtest_xgboost$prresult
predtest_xgboost$prplot
predtest_xgboost$cmresult
predtest_xgboost$cmplot
predtest_xgboost$metrics
predtest_xgboost$diycutoff

# pROC包auc值及其置信区间
pROC::auc(predtest_xgboost$proc)
pROC::ci.auc(predtest_xgboost$proc)

# ROC比较检验
pROC::roc.test(predtrain_xgboost$proc, predtest_xgboost$proc)


# 合并训练集和测试集上ROC曲线
predtrain_xgboost$rocresult %>%
  bind_rows(predtest_xgboost$rocresult) %>%
  mutate(dataAUC = paste(data, " ROCAUC:", round(ROCAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上PR曲线
predtrain_xgboost$prresult %>%
  bind_rows(predtest_xgboost$prresult) %>%
  mutate(dataAUC = paste(data, " PRAUC:", round(PRAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上性能指标
predtrain_xgboost$metrics %>%
  bind_rows(predtest_xgboost$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_xgboost <- bestcv4cls2(
  wkflow = wk_xgboost,
  tuneresult = tune_xgboost,
  hpbest = hpbest_xgboost,
  yname = "AHD",
  modelname = "Xgboost",
  v = 5,
  positivelevel = yourpositivelevel
)
evalcv_xgboost$plotcv
evalcv_xgboost$evalcv

# 保存评估结果
save(datarecipe_xgboost,
     model_xgboost,
     wk_xgboost,
     hpgrid_xgboost,  # 如果采用贝叶斯优化则替换为 param_xgboost
     tune_xgboost,
     predtrain_xgboost,
     predtest_xgboost,
     evalcv_xgboost,
     file = ".\\cls2\\evalresult_xgboost.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_xgboost_heart <- final_xgboost
traindata_heart <- traindata
save(final_xgboost_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_xgboost_heart.RData")

# 预测
predresult <- Heart %>%
  bind_cols(predict(final_xgboost, new_data = Heart, type = "prob"))%>%
  mutate(
    .pred_class = factor(
      ifelse(.pred_Yes >= predtrain_xgboost$diycutoff, 
             yourpositivelevel, 
             yournegativelevel)
    )
  )
# readr::write_excel_csv(predresult, "Xgboost二分类预测结果.csv")

###################################################################

# 自变量数据集
colnames(traindata)
traindatax <- traindata %>%
  dplyr::select(-AHD)
colnames(traindatax)

# 提取最终的算法模型
final_xgboost2 <- final_xgboost %>%
  extract_fit_engine()
final_xgboost2

# 变量重要性
importance_matrix <- 
  xgboost::xgb.importance(model = final_xgboost2)
print(importance_matrix)
xgboost::xgb.plot.importance(
  importance_matrix = importance_matrix,
  measure = "Cover",
  col = "skyblue"
)
# SHAP
xgboost::xgb.plot.shap(
  data = as.matrix(final_xgboost %>%
                     extract_recipe() %>%
                     bake(new_data = traindata) %>%
                     dplyr::select(-AHD)), 
  model = final_xgboost2,
  top_n = 5
)

######################## DALEX解释对象

explainer_xgboost <- DALEXtra::explain_tidymodels(
  final_xgboost, 
  data = traindatax,
  y = ifelse(traindata$AHD == yourpositivelevel, 1, 0),
  type = "classification",
  label = "Xgboost"
)
# 变量重要性
set.seed(42)
vip_xgboost <- DALEX::model_parts(
  explainer_xgboost,
  type = "ratio"
)
plot(vip_xgboost)
plot(vip_xgboost, show_boxplots = FALSE)

# 变量偏依赖图
# 连续型变量
set.seed(42)
pdpc_xgboost <- DALEX::model_profile(
  explainer_xgboost,
  variables = colnames(traindatax)
)
plot(pdpc_xgboost)
# 分类变量
set.seed(42)
pdpd_xgboost <- DALEX::model_profile(
  explainer_xgboost,
  variables = colnames(traindatax)[c(2,3,6,7,9,11,13)]  # 分类变量所在位置
)
plot(pdpd_xgboost)

# 单样本预测分解
set.seed(42)
shap_xgboost <- DALEX::predict_parts(
  explainer = explainer_xgboost, 
  new_observation = traindatax[1, ], 
  type = "shap"
)
plot(shap_xgboost, 
     max_features = ncol(traindatax))
plot(shap_xgboost, 
     max_features = ncol(traindatax), 
     show_boxplots = FALSE)

######################## fastshap包

shapresult <- shap4cls2(
  finalmodel = final_xgboost,
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
  baseline = mean(predtrain_xgboost$prediction$.pred_Yes),
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

#################################################################

# 学习曲线
lcN <- 
  floor(seq(nrow(traindata)%/%2, nrow(traindata), length = 10))
lcresult_xgboost <- data.frame()
for (i in lcN) {
  
  set.seed(i)
  traindatai <- traindata[sample(nrow(traindata), i), ]
  
  i_xgboost <-  wk_xgboost %>%
    finalize_workflow(hpbest_xgboost) %>%
    fit(traindatai)
  
  predtrain_i_xgboost <- eval4cls2(
    model = i_xgboost, 
    dataset = traindatai, 
    yname = "AHD", 
    modelname = "Xgboost", 
    datasetname = paste("traindata", i, sep = "-"),
    cutoff = "yueden",
    positivelevel = yourpositivelevel,
    negativelevel = yournegativelevel
  )
  
  predtest_i_xgboost <- eval4cls2(
    model = i_xgboost, 
    dataset = testdata, 
    yname = "AHD", 
    modelname = "Xgboost", 
    datasetname = paste("testdata", i, sep = "-"),
    cutoff = predtrain_i_xgboost$diycutoff,
    positivelevel = yourpositivelevel,
    negativelevel = yournegativelevel
  )
  
  predi <- bind_rows(predtrain_i_xgboost$metrics, predtest_i_xgboost$metrics)
  lcresult_xgboost <- rbind(lcresult_xgboost, predi)
}
# 图示
lcresult_xgboost %>%
  separate(data, into = c("dataset", "N"), sep = "-") %>%
  mutate(N = as.numeric(N),
         dataset = forcats::as_factor(dataset)) %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = N, y = .estimate, color = dataset)) +
  geom_point() +
  geom_smooth(se = F) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "Samples in traindata", y = "ROCAUC", color = "") +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())


