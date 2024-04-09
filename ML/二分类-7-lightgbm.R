# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类---lightgbm

# https://www.tidymodels.org/find/parsnip/
# https://parsnip.tidymodels.org/reference/boost_tree.html
# https://parsnip.tidymodels.org/reference/details_boost_tree_lightgbm.html

# 模型评估指标
# https://cran.r-project.org/web/packages/yardstick/vignettes/metric-types.html

##############################################################

# install.packages("tidymodels")
library(tidymodels)
library(bonsai)
source("tidyfuncs4cls2.R")

# 读取数据
Heart <- readr::read_csv(file.choose())
colnames(Heart) 

Heart <- X1
# 修正变量类型
# 将分类变量转换为factor
for(i in c(8)){ 
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
datarecipe_lightgbm <- recipe(AHD ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), 
             naming = new_dummy_names) %>% 
  prep()
datarecipe_lightgbm


# 设定模型
model_lightgbm <- boost_tree(
  mode = "classification",
  engine = "lightgbm",
  tree_depth = tune(),
  trees = tune(),
  learn_rate = tune(),
  mtry = tune(),
  min_n = tune(),
  loss_reduction = tune()
)
model_lightgbm

# workflow
wk_lightgbm <- 
  workflow() %>%
  add_recipe(datarecipe_lightgbm) %>%
  add_model(model_lightgbm)
wk_lightgbm

##############################################################

############################  超参数寻优1-网格搜索

# 超参数寻优网格
set.seed(42)
hpgrid_lightgbm <- parameters(
  tree_depth(range = c(1, 3)),
  trees(range = c(100, 500)),
  learn_rate(range = c(-3, -1)),
  mtry(range = c(2, 8)),
  min_n(range = c(5, 10)),
  loss_reduction(range = c(-3, 0))
) %>%
  # grid_regular(levels = c(3, 2, 2, 3, 2, 2)) # 常规网格
  grid_random(size = 5) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_lightgbm
# 网格也可以自己手动生成expand.grid()

# 交叉验证网格搜索过程
set.seed(42)
tune_lightgbm <- wk_lightgbm %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_lightgbm,
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
param_lightgbm <- model_lightgbm %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(2, 10)))

# 贝叶斯优化超参数
set.seed(42)
tune_lightgbm <- wk_lightgbm %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    param_info = param_lightgbm,
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
autoplot(tune_lightgbm)
eval_tune_lightgbm <- tune_lightgbm %>%
  collect_metrics()
eval_tune_lightgbm

# 经过交叉验证得到的最优超参数
hpbest_lightgbm <- tune_lightgbm %>%
  select_by_one_std_err(metric = "roc_auc", desc(min_n))
hpbest_lightgbm

# 采用最优超参数组合训练最终模型
final_lightgbm <- wk_lightgbm %>%
  finalize_workflow(hpbest_lightgbm) %>%
  fit(traindata)
final_lightgbm

##################################################################

# 训练集预测评估
predtrain_lightgbm <- eval4cls2(
  model = final_lightgbm, 
  dataset = traindata, 
  yname = "AHD", 
  modelname = "Lightgbm", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_lightgbm$prediction
predtrain_lightgbm$predprobplot
predtrain_lightgbm$rocresult
predtrain_lightgbm$rocplot
predtrain_lightgbm$prresult
predtrain_lightgbm$prplot
predtrain_lightgbm$cmresult
predtrain_lightgbm$cmplot
predtrain_lightgbm$metrics
predtrain_lightgbm$diycutoff

# pROC包auc值及其置信区间
pROC::auc(predtrain_lightgbm$proc)
pROC::ci.auc(predtrain_lightgbm$proc)

# 预测评估测试集预测评估
predtest_lightgbm <- eval4cls2(
  model = final_lightgbm, 
  dataset = testdata, 
  yname = "AHD", 
  modelname = "Lightgbm", 
  datasetname = "testdata",
  cutoff = predtrain_lightgbm$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_lightgbm$prediction
predtest_lightgbm$predprobplot
predtest_lightgbm$rocresult
predtest_lightgbm$rocplot
predtest_lightgbm$prresult
predtest_lightgbm$prplot
predtest_lightgbm$cmresult
predtest_lightgbm$cmplot
predtest_lightgbm$metrics
predtest_lightgbm$diycutoff

# pROC包auc值及其置信区间
pROC::auc(predtest_lightgbm$proc)
pROC::ci.auc(predtest_lightgbm$proc)

# ROC比较检验
pROC::roc.test(predtrain_lightgbm$proc, predtest_lightgbm$proc)

# 合并训练集和测试集上ROC曲线
predtrain_lightgbm$rocresult %>%
  bind_rows(predtest_lightgbm$rocresult) %>%
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
predtrain_lightgbm$prresult %>%
  bind_rows(predtest_lightgbm$prresult) %>%
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
predtrain_lightgbm$metrics %>%
  bind_rows(predtest_lightgbm$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_lightgbm <- bestcv4cls2(
  wkflow = wk_lightgbm,
  tuneresult = tune_lightgbm,
  hpbest = hpbest_lightgbm,
  yname = "AHD",
  modelname = "Lightgbm",
  v = 5,
  positivelevel = yourpositivelevel
)
evalcv_lightgbm$plotcv
evalcv_lightgbm$evalcv

# 保存评估结果
save(datarecipe_lightgbm,
     model_lightgbm,
     wk_lightgbm,
     hpgrid_lightgbm,
     tune_lightgbm,
     predtrain_lightgbm,
     predtest_lightgbm,
     evalcv_lightgbm,
     file = ".\\cls2\\evalresult_lightgbm.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_lightgbm_heart <- final_lightgbm
traindata_heart <- traindata
save(final_lightgbm_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_lightgbm_heart.RData")

# 预测
predresult <- Heart %>%
  bind_cols(predict(final_lightgbm, new_data = Heart, type = "prob"))%>%
  mutate(
    .pred_class = factor(
      ifelse(.pred_Yes >= predtrain_lightgbm$diycutoff, 
             yourpositivelevel, 
             yournegativelevel)
    )
  )
# readr::write_excel_csv(predresult, "Lightgbm二分类预测结果.csv")

###################################################################

# 自变量数据集
colnames(traindata)
traindatax <- traindata %>%
  dplyr::select(-AHD)
colnames(traindatax)

# 提取最终的算法模型
final_lightgbm2 <- final_lightgbm %>%
  extract_fit_engine()
final_lightgbm2

# 保存lightgbm模型比较特殊
model_file <- 
  tempfile(pattern = "lightgbm", tmpdir = ".", fileext = ".txt")
lightgbm::lgb.save(final_lightgbm2, model_file)

# # 加载也需要自己的函数
# load_booster <- lightgbm::lgb.load(file.choose())

# 变量重要性
lightgbm::lgb.importance(final_lightgbm2, percentage = T)
lightgbm::lgb.plot.importance(
  lightgbm::lgb.importance(final_lightgbm2, percentage = T)
)

# 变量对预测的贡献
lightgbm::lgb.interprete(
  final_lightgbm2, 
  as.matrix(final_lightgbm %>%
              extract_recipe() %>%
              bake(new_data = traindata) %>%
              dplyr::select(-AHD)), 
  1:2
)
lightgbm::lgb.plot.interpretation(
  lightgbm::lgb.interprete(
    final_lightgbm2, 
    as.matrix(final_lightgbm %>%
                extract_recipe() %>%
                bake(new_data = traindata) %>%
                dplyr::select(-AHD)),
    1:2
  )[[1]]
)

######################## DALEX解释对象

explainer_lightgbm <- DALEXtra::explain_tidymodels(
  final_lightgbm, 
  data = traindatax,
  y = ifelse(traindata$AHD == yourpositivelevel, 1, 0),
  type = "classification",
  label = "Lightgbm"
)
# 变量重要性
set.seed(42)
vip_lightgbm <- DALEX::model_parts(
  explainer_lightgbm,
  type = "ratio"
)
plot(vip_lightgbm)
plot(vip_lightgbm, show_boxplots = FALSE)

# 变量偏依赖图
# 连续型变量
set.seed(42)
pdpc_lightgbm <- DALEX::model_profile(
  explainer_lightgbm,
  variables = colnames(traindatax)
)
plot(pdpc_lightgbm)
# 分类变量
set.seed(42)
pdpd_lightgbm <- DALEX::model_profile(
  explainer_lightgbm,
  variables = colnames(traindatax)[c(8)]  # 分类变量所在位置
)
plot(pdpd_lightgbm)

# 单样本预测分解
set.seed(42)
shap_lightgbm <- DALEX::predict_parts(
  explainer = explainer_lightgbm, 
  new_observation = traindatax[1, ], 
  type = "shap"
)
plot(shap_lightgbm, 
     max_features = ncol(traindatax))
plot(shap_lightgbm,
     max_features = ncol(traindatax),
     show_boxplots = FALSE)

######################## fastshap包

shapresult <- shap4cls2(
  finalmodel = final_lightgbm,
  predfunc = function(model, newdata) {
    predict(model, newdata, type = "prob") %>%
      select(ends_with(yourpositivelevel)) %>%
      pull()
  },
  datax = traindatax,
  datay = traindata$AHD,
  yname = "AHD",
  flname = colnames(traindatax)[c(8)],
  lxname = colnames(traindatax)[-c(8)]
)

# 基于shap的变量重要性
shapresult$shapvip
# 单样本预测分解
fastshap::force_plot(
  object = shapresult$shapley[1, ], 
  feature_values = as.data.frame(traindatax)[1, ], 
  baseline = mean(predtrain_lightgbm$prediction$.pred_Yes),
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
lcresult_lightgbm <- data.frame()
for (i in lcN) {
  
  set.seed(i)
  traindatai <- traindata[sample(nrow(traindata), i), ]
  
  i_lightgbm <-  wk_lightgbm %>%
    finalize_workflow(hpbest_lightgbm) %>%
    fit(traindatai)
  
  predtrain_i_lightgbm <- eval4cls2(
    model = i_lightgbm, 
    dataset = traindatai, 
    yname = "AHD", 
    modelname = "Lightgbm", 
    datasetname = paste("traindata", i, sep = "-"),
    cutoff = "yueden",
    positivelevel = yourpositivelevel,
    negativelevel = yournegativelevel
  )
  
  predtest_i_lightgbm <- eval4cls2(
    model = i_lightgbm, 
    dataset = testdata, 
    yname = "AHD", 
    modelname = "Lightgbm", 
    datasetname = paste("testdata", i, sep = "-"),
    cutoff = predtrain_i_lightgbm$diycutoff,
    positivelevel = yourpositivelevel,
    negativelevel = yournegativelevel
  )
  
  predi <- bind_rows(predtrain_i_lightgbm$metrics, predtest_i_lightgbm$metrics)
  lcresult_lightgbm <- rbind(lcresult_lightgbm, predi)
}
# 图示
lcresult_lightgbm %>%
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


