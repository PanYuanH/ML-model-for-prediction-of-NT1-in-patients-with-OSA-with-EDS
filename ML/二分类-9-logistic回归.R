# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类---logistic回归

# https://www.tidymodels.org/find/parsnip/
# https://parsnip.tidymodels.org/reference/logistic_reg.html
# https://parsnip.tidymodels.org/reference/details_logistic_reg_glm.html

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
for(i in c(1,2)){ 
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
datarecipe_logistic <- recipe(AHD ~ ., traindata) %>%
  prep()
datarecipe_logistic


# 设定模型
model_logistic <- logistic_reg(
  mode = "classification",
  engine = "glm"
)
model_logistic

# workflow
wk_logistic <- 
  workflow() %>%
  add_recipe(datarecipe_logistic) %>%
  add_model(model_logistic)
wk_logistic

# 训练模型
final_logistic <- wk_logistic %>%
  fit(traindata)
final_logistic

##################################################################

# 训练集预测评估
predtrain_logistic <- eval4cls2(
  model = final_logistic, 
  dataset = traindata, 
  yname = "AHD", 
  modelname = "Logistic", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_logistic$prediction
predtrain_logistic$predprobplot
predtrain_logistic$rocresult
predtrain_logistic$rocplot
predtrain_logistic$prresult
predtrain_logistic$prplot
predtrain_logistic$cmresult
predtrain_logistic$cmplot
predtrain_logistic$metrics
predtrain_logistic$diycutoff

# pROC包auc值及其置信区间
pROC::auc(predtrain_logistic$proc)
pROC::ci.auc(predtrain_logistic$proc)

# 预测评估测试集预测评估
predtest_logistic <- eval4cls2(
  model = final_logistic, 
  dataset = testdata, 
  yname = "AHD", 
  modelname = "Logistic", 
  datasetname = "testdata",
  cutoff = predtrain_logistic$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_logistic$prediction
predtest_logistic$predprobplot
predtest_logistic$rocresult
predtest_logistic$rocplot
predtest_logistic$prresult
predtest_logistic$prplot
predtest_logistic$cmresult
predtest_logistic$cmplot
predtest_logistic$metrics
predtest_logistic$diycutoff

# pROC包auc值及其置信区间
pROC::auc(predtest_logistic$proc)
pROC::ci.auc(predtest_logistic$proc)

# ROC比较检验
pROC::roc.test(predtrain_logistic$proc, predtest_logistic$proc)

# 合并训练集和测试集上ROC曲线
predtrain_logistic$rocresult %>%
  bind_rows(predtest_logistic$rocresult) %>%
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
predtrain_logistic$prresult %>%
  bind_rows(predtest_logistic$prresult) %>%
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
predtrain_logistic$metrics %>%
  bind_rows(predtest_logistic$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

##################################################################

# 交叉验证
set.seed(42)
cv_logistic <- 
  wk_logistic %>%
  fit_resamples(
    folds,
    metrics = metric_set(yardstick::accuracy, 
                         yardstick::roc_auc, 
                         yardstick::pr_auc),
    control = control_resamples(save_pred = T,
                                verbose = T,
                                event_level = "second",
                                parallel_over = "everything",
                                save_workflow = T)
  )
cv_logistic

# 交叉验证指标结果
evalcv_logistic <- collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  roc_auc(AHD, .pred_Yes, event_level = "second") %>%
  mutate(model = "logistic",
         mean = mean(.estimate),
         sd = sd(.estimate)/sqrt(length(folds$splits)))
evalcv_logistic

# 交叉验证预测结果图示
collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  roc_curve(AHD, .pred_Yes, event_level = "second") %>%
  ungroup() %>%
  left_join(evalcv_logistic, by = "id") %>%
  mutate(idAUC = paste(id, " AUC:", round(.estimate, 4)),
         idAUC = forcats::as_factor(idAUC)) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = idAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())



##################################################################

# 保存评估结果
save(datarecipe_logistic,
     model_logistic,
     wk_logistic,
     cv_logistic,
     predtrain_logistic,
     predtest_logistic,
     evalcv_logistic,
     file = ".\\cls2\\evalresult_logistic.RData")

# 保存模型结果供shiny部署之用，本课程不包括shiny内容
final_logistic_heart <- final_logistic
traindata_heart <- traindata
save(final_logistic_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_logistic_heart.RData")

# 预测
predresult <- Heart %>%
  bind_cols(predict(final_logistic, new_data = Heart, type = "prob"))%>%
  mutate(
    .pred_class = factor(
      ifelse(.pred_Yes >= predtrain_logistic$diycutoff, 
             yourpositivelevel, 
             yournegativelevel)
    )
  )
# readr::write_excel_csv(predresult, "Logistic二分类预测结果.csv")

###################################################################

# 自变量数据集
colnames(traindata)
traindatax <- traindata %>%
  dplyr::select(-AHD)
colnames(traindatax)

# 提取最终的算法模型
final_logistic2 <- final_logistic %>%
  extract_fit_engine()
final_logistic2


######################## DALEX解释对象

explainer_logistic <- DALEXtra::explain_tidymodels(
  final_logistic, 
  data = traindatax,
  y = ifelse(traindata$AHD == yourpositivelevel, 1, 0),
  type = "classification",
  label = "Logistic"
)
# 变量重要性
set.seed(42)
vip_logistic <- DALEX::model_parts(
  explainer_logistic,
  type = "ratio"
)
plot(vip_logistic)
plot(vip_logistic, show_boxplots = FALSE)

# 变量偏依赖图
# 连续型变量
set.seed(42)
pdpc_logistic <- DALEX::model_profile(
  explainer_logistic,
  variables = colnames(traindatax)
)
plot(pdpc_logistic)
# 分类变量
set.seed(42)
pdpd_logistic <- DALEX::model_profile(
  explainer_logistic,
  variables = colnames(traindatax)[c(2,3,6,7,9,11,13)]  # 分类变量所在位置
)
plot(pdpd_logistic)

# 单样本预测分解
set.seed(42)
shap_logistic <- DALEX::predict_parts(
  explainer = explainer_logistic, 
  new_observation = traindatax[1, ], 
  type = "shap"
)
plot(shap_logistic, 
     max_features = ncol(traindatax))
plot(shap_logistic, 
     max_features = ncol(traindatax), 
     show_boxplots = FALSE)

######################## fastshap包

shapresult <- shap4cls2(
  finalmodel = final_logistic,
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
  baseline = mean(predtrain_logistic$prediction$.pred_Yes),
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
lcresult_logistic <- data.frame()
for (i in lcN) {
  
  set.seed(i)
  traindatai <- traindata[sample(nrow(traindata), i), ]
  
  i_logistic <-  wk_logistic %>%
    fit(traindatai)
  
  predtrain_i_logistic <- eval4cls2(
    model = i_logistic, 
    dataset = traindatai, 
    yname = "AHD", 
    modelname = "Logistic", 
    datasetname = paste("traindata", i, sep = "-"),
    cutoff = "yueden",
    positivelevel = yourpositivelevel,
    negativelevel = yournegativelevel
  )
  
  predtest_i_logistic <- eval4cls2(
    model = i_logistic, 
    dataset = testdata, 
    yname = "AHD", 
    modelname = "Logistic", 
    datasetname = paste("testdata", i, sep = "-"),
    cutoff = predtrain_i_logistic$diycutoff,
    positivelevel = yourpositivelevel,
    negativelevel = yournegativelevel
  )
  
  predi <- bind_rows(predtrain_i_logistic$metrics, predtest_i_logistic$metrics)
  lcresult_logistic <- rbind(lcresult_logistic, predi)
}
# 图示
lcresult_logistic %>%
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


