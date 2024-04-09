# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类---LASSO岭回归弹性网络

# https://www.tidymodels.org/find/parsnip/
# https://parsnip.tidymodels.org/reference/logistic_reg.html
# https://parsnip.tidymodels.org/reference/details_logistic_reg_glmnet.html

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
datarecipe_enet <- recipe(AHD ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), 
             naming = new_dummy_names) %>% 
  step_normalize(all_predictors()) %>%
  prep()
datarecipe_enet


# 设定模型
model_enet <- logistic_reg(
  mode = "classification",
  engine = "glmnet",
  # mixture = 1,   # LASSO
  # mixture = 0,  # 岭回归
  mixture = tune(),
  penalty = tune()
)
model_enet

# workflow
wk_enet <- 
  workflow() %>%
  add_recipe(datarecipe_enet) %>%
  add_model(model_enet)
wk_enet

##############################################################

############################  超参数寻优1-网格搜索

# 超参数寻优网格
set.seed(42)
hpgrid_enet <- parameters(
  mixture(),
  penalty(range = c(-5, 0))
) %>%
  grid_regular(levels = c(5, 20)) # 常规网格
# grid_random(size = 5) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_enet
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_enet <- wk_enet %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_enet,
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

# 贝叶斯优化超参数
set.seed(42)
tune_enet <- wk_enet %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
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
autoplot(tune_enet)
eval_tune_enet <- tune_enet %>%
  collect_metrics()
eval_tune_enet

# 经过交叉验证得到的最优超参数
hpbest_enet <- tune_enet %>%
  select_by_one_std_err(metric = "roc_auc", desc(penalty))
hpbest_enet

# 采用最优超参数组合训练最终模型
final_enet <- wk_enet %>%
  finalize_workflow(hpbest_enet) %>%
  fit(traindata)
final_enet

##################################################################

# 训练集预测评估
predtrain_enet <- eval4cls2(
  model = final_enet, 
  dataset = traindata, 
  yname = "AHD", 
  modelname = "弹性网络", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_enet$prediction
predtrain_enet$predprobplot
predtrain_enet$rocresult
predtrain_enet$rocplot
predtrain_enet$prresult
predtrain_enet$prplot
predtrain_enet$cmresult
predtrain_enet$cmplot
predtrain_enet$metrics
predtrain_enet$diycutoff

# pROC包auc值及其置信区间
pROC::auc(predtrain_enet$proc)
pROC::ci.auc(predtrain_enet$proc)

# 预测评估测试集预测评估
predtest_enet <- eval4cls2(
  model = final_enet, 
  dataset = testdata, 
  yname = "AHD", 
  modelname = "弹性网络", 
  datasetname = "testdata",
  cutoff = predtrain_enet$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_enet$prediction
predtest_enet$predprobplot
predtest_enet$rocresult
predtest_enet$rocplot
predtest_enet$prresult
predtest_enet$prplot
predtest_enet$cmresult
predtest_enet$cmplot
predtest_enet$metrics
predtest_enet$diycutoff

# pROC包auc值及其置信区间
pROC::auc(predtest_enet$proc)
pROC::ci.auc(predtest_enet$proc)

# ROC比较检验
pROC::roc.test(predtrain_enet$proc, predtest_enet$proc)


# 合并训练集和测试集上ROC曲线
predtrain_enet$rocresult %>%
  bind_rows(predtest_enet$rocresult) %>%
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
predtrain_enet$prresult %>%
  bind_rows(predtest_enet$prresult) %>%
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
predtrain_enet$metrics %>%
  bind_rows(predtest_enet$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_enet <- bestcv4cls2(
  wkflow = wk_enet,
  tuneresult = tune_enet,
  hpbest = hpbest_enet,
  yname = "AHD",
  modelname = "弹性网络",
  v = 5,
  positivelevel = yourpositivelevel
)
evalcv_enet$plotcv
evalcv_enet$evalcv

# 保存评估结果
save(datarecipe_enet,
     model_enet,
     wk_enet,
     hpgrid_enet,   # 如果采用贝叶斯优化则无需保存
     tune_enet,
     predtrain_enet,
     predtest_enet,
     evalcv_enet,
     file = ".\\cls2\\evalresult_enet.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_enet_heart <- final_enet
traindata_heart <- traindata
save(final_enet_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_enet_heart.RData")

# 预测
predresult <- Heart %>%
  bind_cols(predict(final_enet, new_data = Heart, type = "prob"))%>%
  mutate(
    .pred_class = factor(
      ifelse(.pred_Yes >= predtrain_enet$diycutoff, 
             yourpositivelevel, 
             yournegativelevel)
    )
  )
# readr::write_excel_csv(predresult, "弹性网络二分类预测结果.csv")

###################################################################

# 自变量数据集
colnames(traindata)
traindatax <- traindata %>%
  dplyr::select(-AHD)
colnames(traindatax)

# 提取最终的算法模型
final_enet2 <- final_enet %>%
  extract_fit_engine()
final_enet2

# 非零系数自变量
tidy(final_enet) %>%
  filter(term != "(Intercept)", estimate != 0) %>%
  pull(term)

######################## DALEX解释对象

explainer_enet <- DALEXtra::explain_tidymodels(
  final_enet, 
  data = traindatax,
  y = ifelse(traindata$AHD == yourpositivelevel, 1, 0),
  type = "classification",
  label = "弹性网络"
)
# 变量重要性
set.seed(42)
vip_enet <- DALEX::model_parts(
  explainer_enet,
  type = "ratio"
)
plot(vip_enet)
plot(vip_enet, show_boxplots = FALSE)

# 变量偏依赖图
# 连续型变量
set.seed(42)
pdpc_enet <- DALEX::model_profile(
  explainer_enet,
  variables = colnames(traindatax)
)
plot(pdpc_enet)
# 分类变量
set.seed(42)
pdpd_enet <- DALEX::model_profile(
  explainer_enet,
  variables = colnames(traindatax)[c(2,3,6,7,9,11,13)]  # 分类变量所在位置
)
plot(pdpd_enet)

# 单样本预测分解
set.seed(42)
shap_enet <- DALEX::predict_parts(
  explainer = explainer_enet, 
  new_observation = traindatax[1, ], 
  type = "shap"
)
plot(shap_enet, 
     max_features = ncol(traindatax))
plot(shap_enet, 
     max_features = ncol(traindatax),
     show_boxplots = FALSE)

######################## fastshap包

shapresult <- shap4cls2(
  finalmodel = final_enet,
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
  baseline = mean(predtrain_enet$prediction$.pred_Yes),
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
lcresult_enet <- data.frame()
for (i in lcN) {
  
  set.seed(i)
  traindatai <- traindata[sample(nrow(traindata), i), ]
  
  i_enet <-  wk_enet %>%
    finalize_workflow(hpbest_enet) %>%
    fit(traindatai)
  
  predtrain_i_enet <- eval4cls2(
    model = i_enet, 
    dataset = traindatai, 
    yname = "AHD", 
    modelname = "弹性网络", 
    datasetname = paste("traindata", i, sep = "-"),
    cutoff = "yueden",
    positivelevel = yourpositivelevel,
    negativelevel = yournegativelevel
  )
  
  predtest_i_enet <- eval4cls2(
    model = i_enet, 
    dataset = testdata, 
    yname = "AHD", 
    modelname = "弹性网络", 
    datasetname = paste("testdata", i, sep = "-"),
    cutoff = predtrain_i_enet$diycutoff,
    positivelevel = yourpositivelevel,
    negativelevel = yournegativelevel
  )
  
  predi <- bind_rows(predtrain_i_enet$metrics, predtest_i_enet$metrics)
  lcresult_enet <- rbind(lcresult_enet, predi)
}
# 图示
lcresult_enet %>%
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


