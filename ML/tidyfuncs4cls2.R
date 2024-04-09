# 多核并行
library(doParallel)
registerDoParallel(
  makePSOCKcluster(
    max(1, (parallel::detectCores(logical = F))-1)
  )
)

###################################################################

# 适用于独热编码过程的新命名函数
new_dummy_names <- function (var, lvl, ordinal = FALSE) {
  args <- vctrs::vec_recycle_common(var, lvl)
  var <- args[[1]]
  lvl <- args[[2]]
  nms <- paste(var, lvl, sep = "_")
  nms
}

###################################################################

# 二分类模型预测评估函数
eval4cls2 <- function(model, dataset, yname, modelname, datasetname,
                      cutoff, positivelevel, negativelevel) {
  
  # 预测概率
  predresult <- model %>%
    predict(new_data = dataset, type = "prob") %>%
    mutate(.obs = dataset[[yname]],
           data = datasetname,
           model = modelname)
  # 预测概率分布
  predprobplot <- predresult %>%
    ggplot(aes(x = .data[[paste0(".pred_", positivelevel)]], 
                      color = .data[[".obs"]])) +
    geom_density(linewidth = 1.2) +
    theme_bw()
  
  # ROC
  predprob4pos <- predresult %>%
    dplyr::select(ends_with(positivelevel)) %>%
    pull()
  rocresult <- predresult %>%
    roc_curve(.obs, 
              ends_with(positivelevel), 
              event_level = "second") %>%
    mutate(data = datasetname,
           model = modelname)
  rocaucresult <- predresult %>%
    roc_auc(.obs, 
            ends_with(positivelevel), 
            event_level = "second")
  rocplot <- autoplot(rocresult) +
    geom_text(
      x = 0.5, 
      y = 0.5,
      label = paste0("ROCAUC=", round(rocaucresult$.estimate, 4))
    ) +
    scale_x_continuous(expand = c(0,0), limits = c(0, 1)) +
    scale_y_continuous(expand = c(0,0), limits = c(0, 1))
  # pROC包
  proc <- pROC::roc(response = dataset[[yname]], 
                    predictor = predprob4pos,
                    levels = c(negativelevel, positivelevel),
                    direction = "<")
  
  # PR
  prresult <- predresult %>%
    pr_curve(.obs, 
             ends_with(positivelevel), 
             event_level = "second") %>%
    mutate(data = datasetname,
           model = modelname)
  praucresult <- predresult %>%
    pr_auc(.obs, 
           ends_with(positivelevel), 
           event_level = "second")
  prplot <- autoplot(prresult) +
    geom_text(
      x = 0.5, 
      y = 0.5,
      label = paste0("PRAUC=", round(praucresult$.estimate, 4))
    ) +
    scale_x_continuous(expand = c(0,0), limits = c(0, 1)) +
    scale_y_continuous(expand = c(0,0), limits = c(0, 1))
  
  # 预测分类
  if (cutoff == "yueden") {
    yuedencf <- rocresult %>%
      mutate(yueden = sensitivity + specificity - 1) %>%
      slice_max(yueden) %>%
      slice_max(sensitivity) %>%
      slice_max(specificity) %>%
      slice_min(.threshold - 0.5) %>%
      # slice_min(0.5 - .threshold) %>%
      pull(.threshold)
  } else {
    yuedencf <- cutoff
  }
  predresult <- predresult %>%
    mutate(
      .pred_class = factor(
        ifelse(predprob4pos >= yuedencf, 
               positivelevel, 
               negativelevel),
        levels = c(negativelevel, positivelevel)
      )
    )
  
  # 混淆矩阵
  cmresult <- predresult %>%
    conf_mat(truth = .obs, estimate = .pred_class)
  cmplot <- autoplot(cmresult, type = "heatmap") +
    scale_fill_gradient(low = "white", high = "skyblue") +
    theme_minimal() +
    theme(text = element_text(size = 15),
          legend.position = "none")
  
  # 合并指标
  evalresult <- cmresult %>%
    summary(event_level = "second") %>%
    bind_rows(rocaucresult) %>%
    bind_rows(praucresult) %>%
    mutate(data = datasetname,
           model = modelname,
           diycutoff = yuedencf)
 
  
  # 返回结果list
  return(list(prediction = predresult,
              predprobplot = predprobplot,
              rocresult = rocresult %>%
                mutate(ROCAUC = rocaucresult$.estimate),
              rocplot = rocplot,
              proc = proc,
              prresult = prresult %>%
                mutate(PRAUC = praucresult$.estimate),
              prplot = prplot,
              cmresult = cmresult,
              cmplot = cmplot,
              metrics = evalresult,
              diycutoff = yuedencf))
}

###################################################################

# 最优超参数交叉验证结果提取函数
bestcv4cls2 <- function(wkflow, tuneresult, hpbest, yname, 
                        modelname, v, positivelevel) {
  
  # 调优的超参数个数
  hplength <- wkflow %>%
    extract_parameter_set_dials() %>%
    pull(name) %>%
    length()
  
  # 交叉验证过程中验证集的预测结果-最优参数
  predcv <- tuneresult %>%
    collect_predictions() %>%
    inner_join(hpbest[, 1:hplength]) %>%
    rename(".obs" = all_of(yname))
  
  # 交叉验证过程中验证集的预测结果评估-最优参数
  evalcv <- predcv %>%
    group_by(id) %>%
    roc_auc(.obs, 
            ends_with(positivelevel), 
            event_level = "second") %>%
    mutate(model = modelname) %>%
    group_by(.metric) %>%
    mutate(mean = mean(.estimate),
           sd = sd(.estimate)/sqrt(v))
  
  # 交叉验证过程中验证集的预测结果图示-最优参数
  plotcv <- predcv %>%
    group_by(id) %>%
    roc_curve(.obs, 
              ends_with(positivelevel), 
              event_level = "second") %>%
    ungroup() %>%
    left_join(evalcv[, c(1,4)]) %>%
    mutate(idAUC = paste(id, " AUC:", round(.estimate, 4)),
           idAUC = forcats::as_factor(idAUC)) %>%
    ggplot(aes(x = 1-specificity, y = sensitivity, color = idAUC)) +
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
  
  return(list(plotcv = plotcv,
              evalcv = evalcv))
}

###################################################################

# shap相关函数
shap4cls2 <- function(finalmodel, predfunc, datax, datay,
                      yname, flname, lxname) {
  
  # shap-explain对象
  set.seed(2024)
  shap <- fastshap::explain(
    finalmodel, 
    X = as.data.frame(datax),
    nsim = 10,
    adjust = T,
    pred_wrapper = predfunc
  )
  
  # 分类变量
  data3d <- NULL
  shapimpd <- NULL
  shapplotd_facet <- NULL
  shapplotd_one <- NULL
  if(!is.null(flname)){
    data1d <- shap %>%
      as.data.frame() %>%
      select(all_of(flname)) %>%  
      dplyr::mutate(id = 1:n()) %>%
      pivot_longer(cols = -ncol(.),
                   names_to = "feature",
                   values_to = "shap")
    data2d <- datax  %>%
      select(all_of(flname)) %>%   
      dplyr::mutate(id = 1:n()) %>%
      pivot_longer(cols = -ncol(.),
                   names_to = "feature")
    data3d <- data1d %>%
      left_join(data2d, by = c("id", "feature")) %>%
      left_join(data.frame(id = 1:length(datay), Y = datay),
                by = "id")
    shapimpd <- data1d %>%
      dplyr::group_by(feature) %>%
      dplyr::summarise(shap.abs.mean = mean(abs(shap), na.rm = T)) %>%
      dplyr::arrange(shap.abs.mean) %>%
      dplyr::mutate(feature = forcats::as_factor(feature))
    
    shapplotd_facet <- data3d %>%
      na.omit() %>%
      ggplot(aes(x = value, y = shap)) +
      geom_boxplot(fill = "lightgreen") +
      geom_point(aes(color = Y), alpha= 0.5) + 
      geom_hline(yintercept = 0, color = "grey10") +
      scale_color_viridis_d() +
      labs(x = "", color = yname) +
      facet_wrap(~feature, scales = "free") +
      theme_bw() +
      theme(axis.text.x = element_text(angle = 30, hjust = 1),
            legend.position = "bottom")
    
    library(ggh4x)
    shapplotd_one <- data3d %>%
      na.omit() %>%
      ggplot(aes(x = interaction(value, feature), y = shap)) +
      geom_boxplot(aes(fill = feature), show.legend = F) +
      geom_hline(yintercept = 0, color = "grey10") +
      scale_x_discrete(NULL, guide = "axis_nested") +
      geom_point(aes(color = Y), alpha= 0.5) +
      scale_colour_viridis_d() +
      labs(x = "", colour = yname) + 
      theme_bw() +
      theme(axis.text.x = element_text(angle = 30, hjust = 1),
            legend.position = "bottom")
  }
  
  # 连续变量
  data3c <- NULL
  shapimpc <- NULL
  shapplotc_facet <- NULL
  shapplotc_one <- NULL
  if(!is.null(lxname)){
    data1c <- shap %>%
      as.data.frame() %>%
      select(all_of(lxname)) %>%
      dplyr::mutate(id = 1:n()) %>%
      pivot_longer(cols = -ncol(.),
                   names_to = "feature",
                   values_to = "shap")
    data2c <- datax  %>%
      select(all_of(lxname)) %>%  
      dplyr::mutate(id = 1:n()) %>%
      pivot_longer(cols = -ncol(.),
                   names_to = "feature")
    data3c <- data1c %>%
      left_join(data2c, by = c("id", "feature")) %>%
      left_join(data.frame(id = 1:length(datay), Y = datay),
                by = "id")
    shapimpc <- data1c %>%
      dplyr::group_by(feature) %>%
      dplyr::summarise(shap.abs.mean = mean(abs(shap), na.rm = T)) %>%
      dplyr::arrange(shap.abs.mean) %>%
      dplyr::mutate(feature = forcats::as_factor(feature))
    
    
    shapplotc_facet <- data3c %>%
      na.omit() %>%
      ggplot(aes(x = value, y = shap)) +
      geom_point(aes(color = Y)) +
      geom_smooth(color = "red") +
      geom_hline(yintercept = 0, color = "grey10") +
      scale_color_viridis_d() +
      labs(x = "", color = yname) + 
      facet_wrap(~feature, scales = "free") +
      theme_bw() +
      theme(legend.position = "bottom")
    
    shapplotc_one <- data3c %>%
      na.omit() %>%
      dplyr::group_by(feature) %>%
      dplyr::mutate(
        value = (value - min(value)) / (max(value) - min(value)),
        feature = factor(feature, levels = levels(shapimpc$feature))
      ) %>%
      dplyr::arrange(value) %>%
      dplyr::ungroup() %>%
      ggplot(aes(x = shap, y = feature, color = value)) +
      ggbeeswarm::geom_quasirandom(width = 0.2) +
      scale_color_gradient(
        low = "red", 
        high = "blue", 
        breaks = c(0, 1), 
        labels = c("Low", "High"), 
        guide = guide_colorbar(barwidth = 0.5,
                               barheight = length(lxname), 
                               ticks = F,
                               title.position = "right",
                               title.hjust = 0.5)
      ) +
      labs(x = "SHAP value", color = "Feature value") +
      theme_bw() +
      theme(legend.title = element_text(angle = -90))
    
  }
  
  # shap变量重要性
  shapvip <- shapimpc %>%
    bind_rows(shapimpd) %>%
    dplyr::arrange(shap.abs.mean) %>%
    dplyr::mutate(feature = as.character(feature),
                  feature = forcats::as_factor(feature)) %>%
    ggplot(aes(x = shap.abs.mean, y = feature, fill = shap.abs.mean)) +
    geom_col(show.legend = F) +
    scale_fill_distiller(palette = "YlOrRd", direction = 1) +
    theme_bw()
  
  return(list(shapley = shap,
              shapdatad = data3d,
              shapplotd_one = shapplotd_one,
              shapplotd_facet = shapplotd_facet,
              shapdatac = data3c,
              shapplotc_one = shapplotc_one,
              shapplotc_facet = shapplotc_facet,
              shapvip = shapvip))
}


