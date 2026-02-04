# data_analysis_happiness_modeling.R
# 使用说明：
# 1. 将 data/Happy_Updated.csv 放到项目根目录的 data/ 子目录
# 2. 在 R 中运行本脚本；会创建 figures/ 和 outputs/ 并保存结果
#
# 依赖（建议在项目使用 renv 管理）:
# tidyverse, here, broom, patchwork, glmnet, caret, MASS, car, lmtest, corrplot

library(tidyverse)
library(here)
library(broom)
library(patchwork)
library(glmnet)
library(caret)
library(MASS)
library(car)
library(lmtest)
library(corrplot)

set.seed(123)

# 路径与目录
data_path <- here::here("data", "Happy_Updated.csv")
if (!file.exists(data_path)) stop("找不到数据文件：", data_path)
dir.create(here::here("figures"), showWarnings = FALSE)
dir.create(here::here("outputs"), showWarnings = FALSE)

# 读入数据（把空白视作 NA）
df <- readr::read_csv(data_path, na = c("", "NA"), show_col_types = FALSE)

# 清理：修正 BOM/编码导致的列名问题（如果有）
names(df) <- trimws(names(df))

required_cols <- c("Country","Ladder_score","LGDP","Support","HLE","Freedom","Corruption","Continent")
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop("缺少必需列：", paste(missing_cols, collapse = ", "))
}

# 处理少数缺失值（例如 Oman 的 Corruption 缺失）——这里把缺失放入 NA 并在后续删除
df <- df %>% mutate(Continent = as.factor(Continent))
initial_n <- nrow(df)

# 简单检查与移除全 NA 行（文件末尾可能有空行）
df <- df %>% filter(if_all(everything(), ~ !is.na(.) | cur_column() == "Country" ))
df <- df %>% filter(complete.cases(select(., all_of(required_cols))))
cleaned_n <- nrow(df)
message("初始行数：", initial_n, "；清理后可用行数：", cleaned_n)

# 数值变量
numeric_vars <- c("Ladder_score","LGDP","Support","HLE","Freedom","Corruption")

# 相关矩阵并保存
cor_mat <- cor(df %>% select(all_of(numeric_vars)), use = "pairwise.complete.obs")
png(here::here("figures","correlation_matrix.png"), width = 900, height = 700)
corrplot(cor_mat, method="color", type="upper", tl.cex=1, addCoef.col="white", number.cex=0.8, cl.lim=c(-1,1), diag=FALSE)
dev.off()

# 数据切分
set.seed(42)
train_index <- createDataPartition(df$Ladder_score, p = 0.8, list = FALSE)
train <- df[train_index, ]
test  <- df[-train_index, ]

# 基本公式
base_formula <- Ladder_score ~ LGDP + Support + HLE + Freedom + Corruption

# 训练多个模型
lm_base <- lm(base_formula, data = train)
lm_cont <- lm(update(base_formula, . ~ . + as.factor(Continent)), data = train)
lm_inter <- lm(Ladder_score ~ LGDP * Support + HLE * Freedom + Corruption + as.factor(Continent), data = train)
step_model <- tryCatch({
  MASS::stepAIC(lm_cont, direction = "both", trace = FALSE)
}, error = function(e) lm_cont)
rlm_model <- MASS::rlm(base_formula, data = train)

# glmnet 模型需要矩阵
x_train <- model.matrix(Ladder_score ~ LGDP + Support + HLE + Freedom + Corruption + Continent, data = train)[, -1]
y_train <- train$Ladder_score
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1, nfolds = 5, standardize = TRUE)
lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = cv_lasso$lambda.min, standardize = TRUE)
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0, nfolds = 5, standardize = TRUE)
ridge_model <- glmnet(x_train, y_train, alpha = 0, lambda = cv_ridge$lambda.min, standardize = TRUE)

# 评估函数（测试集）
eval_model <- function(model, newdata, type = c("lm","glmnet","rlm")) {
  type <- match.arg(type)
  if (type %in% c("lm","rlm")) {
    preds <- predict(model, newdata = newdata)
  } else {
    x_new <- model.matrix(Ladder_score ~ LGDP + Support + HLE + Freedom + Corruption + Continent, data = newdata)[, -1]
    preds <- as.numeric(predict(model, newx = x_new, s = model$lambda))
  }
  actuals <- newdata$Ladder_score
  tibble(
    RMSE = sqrt(mean((actuals - preds)^2)),
    MAE  = mean(abs(actuals - preds)),
    R2   = 1 - sum((actuals - preds)^2) / sum((actuals - mean(actuals))^2)
  )
}

# 计算所有模型在测试集上的表现
evals <- list(
  lm_base = eval_model(lm_base, test, "lm"),
  lm_cont = eval_model(lm_cont, test, "lm"),
  lm_inter = eval_model(lm_inter, test, "lm"),
  step_model = eval_model(step_model, test, "lm"),
  rlm_model = eval_model(rlm_model, test, "rlm"),
  lasso = eval_model({ tmp <- lasso_model; tmp$lambda <- cv_lasso$lambda.min; tmp }, test, "glmnet"),
  ridge = eval_model({ tmp <- ridge_model; tmp$lambda <- cv_ridge$lambda.min; tmp }, test, "glmnet")
)
eval_df <- bind_rows(lapply(names(evals), function(n) tibble(Model = n, evals[[n]])), .id = NULL)
write_csv(eval_df, here::here("outputs","model_evaluation_testset.csv"))

# 保存系数（简洁）
save_coeffs <- function(fit, name, type = c("lm","glmnet","rlm")) {
  type <- match.arg(type)
  if (type == "glmnet") {
    coef_mat <- as.matrix(coef(fit, s = fit$lambda))
    dfc <- tibble(term = rownames(coef_mat), estimate = as.numeric(coef_mat[,1]))
  } else {
    dfc <- broom::tidy(fit)
  }
  write_csv(dfc, here::here("outputs", paste0("coefficients_", name, ".csv")))
  invisible(dfc)
}

save_coeffs(lm_base, "lm_base", "lm")
save_coeffs(lm_cont, "lm_cont", "lm")
save_coeffs(lm_inter, "lm_inter", "lm")
save_coeffs(step_model, "step_model", "lm")
save_coeffs(rlm_model, "rlm_model", "rlm")
save_coeffs({ tmp <- lasso_model; tmp$lambda <- cv_lasso$lambda.min; tmp }, "lasso", "glmnet")
save_coeffs({ tmp <- ridge_model; tmp$lambda <- cv_ridge$lambda.min; tmp }, "ridge", "glmnet")

# LASSO 路径图
png(here::here("figures","lasso_path.png"), width = 900, height = 700)
plot(cv_lasso$glmnet.fit, xvar = "lambda", label = TRUE)
abline(v = log(cv_lasso$lambda.min), col = "red", lty = 2)
dev.off()

# RMSE 柱状对比图
p_eval <- ggplot(eval_df, aes(x = reorder(Model, RMSE), y = RMSE, fill = Model)) +
  geom_col(show.legend = FALSE) + coord_flip() +
  labs(title = "模型在测试集上的 RMSE（越低越好）", x = "", y = "RMSE") + theme_minimal()
ggsave(here::here("figures","model_rmse_compare.png"), p_eval, width = 8, height = 4, dpi = 300)

# 选择最佳模型（测试集 RMSE 最低）
best_name <- eval_df %>% arrange(RMSE) %>% slice(1) %>% pull(Model)
best_name
# 将 best_model 指向相应对象
best_model <- switch(best_name,
                     lm_base = lm_base,
                     lm_cont = lm_cont,
                     lm_inter = lm_inter,
                     step_model = step_model,
                     rlm_model = rlm_model,
                     lasso = { tmp <- lasso_model; tmp$lambda <- cv_lasso$lambda.min; tmp },
                     ridge = { tmp <- ridge_model; tmp$lambda <- cv_ridge$lambda.min; tmp })

# 残差诊断（对线性/rlm）
if (inherits(best_model, "lm") || inherits(best_model, "rlm")) {
  png(here::here("figures", paste0("resid_vs_fitted_", best_name, ".png")), width = 900, height = 700)
  plot(best_model, which = 1, main = paste("Residuals vs Fitted -", best_name))
  dev.off()
  png(here::here("figures", paste0("qq_", best_name, ".png")), width = 900, height = 700)
  plot(best_model, which = 2, main = paste("Q-Q Plot -", best_name))
  dev.off()
} else {
  # glmnet -> 基于预测与实际的残差图
  x_test <- model.matrix(Ladder_score ~ LGDP + Support + HLE + Freedom + Corruption + Continent, data = test)[, -1]
  preds <- as.numeric(predict(best_model, newx = x_test, s = best_model$lambda))
  resid <- test$Ladder_score - preds
  png(here::here("figures", paste0("glmnet_resid_vs_pred_", best_name, ".png")), width = 900, height = 700)
  plot(preds, resid, xlab = "Predicted", ylab = "Residuals", main = paste("Residuals vs Predicted -", best_name))
  abline(h = 0, col = "red")
  dev.off()
}

# 自动生成中文摘要（训练集结果示例）
generate_summary_cn <- function(fit, name) {
  if (!inherits(fit, "lm") && !inherits(fit, "rlm")) return(paste0(name, " 非线性回归模型，略过。"))
  s <- summary(fit)
  coefs <- broom::tidy(fit)
  sig <- coefs %>% filter(term != "(Intercept)" & !is.na(p.value) & p.value <= 0.05)
  lines <- c()
  lines <- c(lines, paste0("模型：", name))
  lines <- c(lines, paste0("训练样本量：", nobs(fit)))
  lines <- c(lines, paste0("R-squared: ", round(s$r.squared, 3), "；Adj R-squared: ", round(s$adj.r.squared, 3)))
  if (nrow(sig) == 0) {
    lines <- c(lines, "在 0.05 水平下未发现显著自变量（不含截距）。")
  } else {
    lines <- c(lines, "在 0.05 水平下显著的自变量：")
    for (i in seq_len(nrow(sig))) {
      trm <- sig$term[i]
      est <- round(sig$estimate[i], 3)
      dir <- ifelse(est > 0, "正向", "负向")
      pval <- signif(sig$p.value[i], 3)
      lines <- c(lines, paste0("- ", trm, "：估计 ", est, "（", dir, "，p=", pval, "）"))
    }
  }
  paste(lines, collapse = "\n")
}

sums <- c(
  generate_summary_cn(lm_base, "lm_base"),
  generate_summary_cn(lm_cont, "lm_cont"),
  generate_summary_cn(lm_inter, "lm_inter"),
  generate_summary_cn(step_model, "step_model"),
  generate_summary_cn(rlm_model, "rlm_model")
)
write_lines(sums, here::here("outputs","automated_model_summaries_cn.txt"))

message("R 脚本已执行完毕。请查看 outputs/ 与 figures/ 中的结果文件。")