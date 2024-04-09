# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类---模型比较

#############################################################
# remotes::install_github("tidymodels/probably")
library(tidymodels)

# source("二分类-1-决策树.R")
# source("二分类-2-随机森林.R")
# source("二分类-3-xgboost.R")
# source("二分类-4-lasso岭回归弹性网络.R")
# source("二分类-5-svm.R")
# source("二分类-6-单隐藏层神经网络.R")
# source("二分类-7-lightgbm.R")
# source("二分类-8-KNN.R")
# source("二分类-9-logistic回归.R")

# 加载各个模型的评估结果
evalfiles <- list.files(".\\cls2\\", full.names = T)
lapply(evalfiles, load, .GlobalEnv)

# 横向比较的模型个数
nmodels <- 9
cols4model <- rainbow(nmodels)  # 模型统一配色
#############################################################

# 各个模型在测试集上的误差指标
predtest_dt$metrics
eval <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt, predtest_enet,
              predtest_knn, predtest_lightgbm, predtest_rf,
              predtest_xgboost, predtest_svm, predtest_mlp), 
         "[[", 
         "metrics")
)
eval
# 平行线图
eval_max <-   eval %>% 
  group_by(.metric) %>%
  slice_max(.estimate)
eval_min <-   eval %>% 
  group_by(.metric) %>%
  slice_min(.estimate)

eval %>%
  ggplot(aes(x = .metric, y = .estimate, color = model)) +
  geom_point() +
  geom_line(aes(group = model)) +
  ggrepel::geom_text_repel(eval_max, 
                           mapping = aes(label = model), 
                           nudge_y = 0.05,
                           angle = 90,
                           show.legend = F) +
  ggrepel::geom_text_repel(eval_min, 
                           mapping = aes(label = model), 
                           nudge_y = -0.05,
                           angle = 90,
                           show.legend = F) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# 各个模型在测试集上的误差指标表格
eval2 <- eval %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
eval2

# 各个模型在测试集上的误差指标图示
# ROCAUC
eval2 %>%
  ggplot(aes(x = model, y = roc_auc, fill = model)) +
  geom_col(width = 0.5, show.legend = F) +
  geom_text(aes(label = round(roc_auc, 2)), 
            nudge_y = -0.03) +
  scale_fill_manual(values = cols4model) +
  theme_bw()

#############################################################

# 各个模型在测试集上的预测概率
predtest <- bind_rows(
  lapply(list(predtest_logistic, predtest_dt, predtest_enet,
              predtest_knn, predtest_lightgbm, predtest_rf,
              predtest_xgboost, predtest_svm, predtest_mlp), 
         "[[", 
         "prediction")
)
predtest

# 各个模型在测试集上的ROC
predtest %>%
  group_by(model) %>%
  roc_curve(.obs, .pred_Yes, event_level = "second") %>%
  left_join(eval2[, c("model", "roc_auc")]) %>%
  mutate(modelauc = paste0(model, 
                           ", ROCAUC=", round(roc_auc, 4))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = modelauc)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(color = "", title = paste0("ROCs on testdata")) +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 各个模型在测试集上的PR
predtest %>%
  group_by(model) %>%
  pr_curve(.obs, .pred_Yes, event_level = "second") %>%
  left_join(eval2[, c("model", "pr_auc")]) %>%
  mutate(modelauc = paste0(model, 
                           ", PRAUC=", round(pr_auc, 4))) %>%
  ggplot(aes(x = recall, y = precision, color = modelauc)) +
  geom_path(linewidth = 1, slope = -1, intercept = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_color_manual(values = cols4model) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(color = "", title = paste0("PRs on testdata")) +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())

#############################################################

# 各个模型在测试集上的预测概率---宽数据
predtest2 <- predtest %>%
  select(-.pred_No) %>%
  mutate(id = rep(1:nrow(predtest_logistic$prediction), 
                  length(unique(predtest$model)))) %>%
  pivot_wider(id_cols = c(id, .obs), 
              names_from = model, 
              values_from = .pred_Yes) %>%
  select(id, .obs, sort(unique(predtest$model)))
predtest2

#############################################################


# 各个模型在测试集上的校准曲线
library(PredictABEL)
predtest4 <- predtest2
predtest4$.obs <- ifelse(predtest4$.obs == "No", 0, 1)
calall <- data.frame()
hlresult <- data.frame()
for (i in 3:ncol(predtest4)) {
  cal <- plotCalibration(data = as.data.frame(predtest4),
                         cOutcome = 2,
                         predRisk = predtest4[[i]],
                         groups = 3) # 分组数可以改
  
  hl <- data.frame(model = colnames(predtest4)[i],
                   chisq = cal$Chi_square,
                   df = cal$df,
                   p = cal$p_value)
  hlresult <- rbind(hlresult, hl)
  
  caldf <- cal$Table_HLtest %>%
    as.data.frame() %>%
    rownames_to_column("pi") %>%
    mutate(model = colnames(predtest4)[i])
  calall <- rbind(calall, caldf)
}
hlresult2 <- hlresult %>%
  mutate(meanpred = 0.75,
         meanobs = 0.25,
         text = paste0("HL, p=", round(p,2)))
calall %>%
  ggplot(aes(x = meanpred, y = meanobs)) +
  geom_point(color = "brown1") +
  geom_line(color = "brown1") +
  geom_abline(slope = 1, intercept = 0) +
  geom_text(hlresult2,
            mapping = aes(x = meanpred, y = meanobs, label = text)) +
  facet_wrap(~model) +
  theme_bw()

# 校准曲线附加误差棒
cal_formula <- as.formula(
  paste0(".obs ~ ",
         paste(sort(unique(predtest$model)), collapse = " + "))
)
cal_formula
caret::calibration(cal_formula,
                   data = predtest2,
                   cuts = 3, # 可以改大改小
                   class = "Yes") %>%
  ggplot() +
  geom_line() +
  scale_color_manual(values = cols4model) +
  facet_wrap(~Model) +
  theme_bw() +
  theme(legend.position = "none")

# 校准曲线附加置信区间
library(probably)
# 算法1
predtest %>%
  cal_plot_breaks(.obs, 
                  .pred_Yes, 
                  event_level = "second", 
                  num_breaks = 5,  # 可以改大改小
                  .by = model) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none")
# 算法2
predtest %>%
  cal_plot_windowed(.obs, 
                    .pred_Yes, 
                    event_level = "second", 
                    window_size = 0.5,  # 可以改大改小
                    .by = model) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none")

# brier_score
bs <- predtest %>%
  group_by(model) %>%
  yardstick::brier_class(.obs, .pred_No) %>%
  mutate(meanpred = 0.8,
         meanobs = 0.25,
         text = paste0("BS: ", round(.estimate, 4)))
# 附加bs
predtest %>%
  cal_plot_windowed(.obs, 
                    .pred_Yes, 
                    event_level = "second", 
                    window_size = 0.8,
                    .by = model) +
  geom_text(
    bs,
    mapping = aes(x = meanpred, y = meanobs, label = text)
  ) +
  scale_color_manual(values = cols4model) +
  theme_bw() +
  theme(legend.position = "none")



#############################################################

# 各个模型在测试集上的DCA
dca_obj <- dcurves::dca(as.formula(
  paste0(".obs ~ ", 
         paste(colnames(predtest2)[3:ncol(predtest2)], 
               collapse = " + "))
),
  data = predtest2,
  thresholds = seq(0, 1, by = 0.01)
)
plot(dca_obj, smooth = T, span = 0.5) +
  scale_color_manual(values = c("black", "grey", cols4model)) +
  labs(title = "DCA on testdata")

#############################################################

# 各个模型交叉验证的各折指标点线图
evalcv <- evalcv <- bind_rows(
  evalcv_logistic, 
  lapply(list(evalcv_dt, evalcv_enet, 
              evalcv_knn, evalcv_lightgbm, evalcv_rf, 
              evalcv_xgboost, evalcv_svm, evalcv_mlp), 
         "[[", 
         "evalcv")
)
evalcv

evalcv_max <-   evalcv %>% 
  group_by(id) %>%
  slice_max(.estimate)
evalcv_min <-   evalcv %>% 
  group_by(id) %>%
  slice_min(.estimate)

evalcv %>%
  ggplot(aes(x = id, y = .estimate, 
             group = model, color = model)) +
  geom_point() +
  geom_line() +
  ggrepel::geom_text_repel(evalcv_max, 
                           mapping = aes(label = model), 
                           nudge_y = 0.01,
                           show.legend = F) +
  ggrepel::geom_text_repel(evalcv_min, 
                           mapping = aes(label = model), 
                           nudge_y = -0.01,
                           show.legend = F) +
  scale_y_continuous(limits = c(0.5, 1)) +
  scale_color_manual(values = cols4model) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# 各个模型交叉验证的指标平均值图(带上下限)
evalcv %>%
  group_by(model) %>%
  sample_n(size = 1) %>%
  ungroup() %>%
  ggplot(aes(x = model, y = mean, color = model)) +
  geom_point(size = 2, show.legend = F) +
  # geom_line(group = 1) +
  geom_errorbar(aes(ymin = mean-sd, 
                    ymax = mean+sd),
                width = 0.1, 
                linewidth = 1.2,
                show.legend = F) +
  scale_y_continuous(limits = c(0.7, 1)) +
  scale_color_manual(values = cols4model) +
  labs(y = "cv roc_auc") +
  theme_bw()




