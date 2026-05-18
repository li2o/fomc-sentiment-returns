############################################################
# Thesis data analysis
# Federal Reserve communication sentiment and asset returns
############################################################

# -----------------------------
# 0. Packages
# -----------------------------

required_packages <- c(
  "tidyverse",
  "lubridate",
  "broom",
  "sandwich",
  "lmtest",
  "zoo",
  "modelsummary",
  "ragg",
  "systemfonts",
  "patchwork"
)

new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(new_packages) > 0) {
  install.packages(new_packages)
}

library(tidyverse)
library(lubridate)
library(broom)
library(sandwich)
library(lmtest)
library(modelsummary)
library(ragg)
library(systemfonts)
library(patchwork)


# -----------------------------
# 1. File paths
# -----------------------------

minutes_path <- "data/master_dataset_minutes_reduced.csv"
press_path   <- "data/master_dataset_press_conferences_reduced.csv"
fomc_document_level_path <- "llm_analysis/outputs/document_level/fomc_document_level.csv"
press_document_level_path <- "llm_analysis/outputs/document_level/press_conferences_document_level.csv"

output_dir <- file.path("analysis", "outputs")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

plot_font_candidates <- c(
  "CM Roman",
  "Computer Modern Roman",
  "CMU Serif",
  "Latin Modern Roman"
)

available_fonts <- unique(systemfonts::system_fonts()$family)
plot_font_family <- plot_font_candidates[plot_font_candidates %in% available_fonts][1]

if (is.na(plot_font_family)) {
  plot_font_family <- "serif"
  warning(
    "CM Roman / Computer Modern is not installed as a system font. ",
    "Plots will use the default serif font until CM Roman is installed."
  )
}

theme_analysis <- function() {
  theme_minimal(base_family = plot_font_family)
}

research_colors <- list(
  text = "#222222",
  grid = "#E6E6E6",
  reference = "#777777",
  primary = "#003D5C",
  primary_light = "#D6E3EA",
  main = c(
    "#003D5C",
    "#2CA58D",
    "#BC4C96",
    "#FF5F66",
    "#FFA600"
  ),
  document_type = c(
    "Policy statements" = "#003D5C",
    "Minutes" = "#BC4C96",
    "Press conferences" = "#FFA600"
  ),
  agent = c(
    "Households" = "#003D5C",
    "Firms" = "#2CA58D",
    "Financial Sector" = "#BC4C96",
    "Government" = "#FF5F66",
    "Central Bank" = "#FFA600"
  )
)

save_plot <- function(filename_base, plot, width, height, dpi = 300) {
  ggsave(
    filename = file.path(output_dir, paste0(filename_base, ".pdf")),
    plot = plot,
    width = width,
    height = height,
    device = grDevices::cairo_pdf,
    family = plot_font_family
  )

  ggsave(
    filename = file.path(output_dir, paste0(filename_base, ".png")),
    plot = plot,
    width = width,
    height = height,
    dpi = dpi,
    device = ragg::agg_png
  )
}


# -----------------------------
# 2. Load data
# -----------------------------

minutes <- read_csv(minutes_path, show_col_types = FALSE) %>%
  mutate(
    Date = as.Date(Date),
    communication_type = "Minutes"
  )

press <- read_csv(press_path, show_col_types = FALSE) %>%
  mutate(
    Date = as.Date(Date),
    communication_type = "Press conferences"
  )

df_all <- bind_rows(minutes, press)

document_sentence_counts <- bind_rows(
  read_csv(fomc_document_level_path, show_col_types = FALSE) %>%
    filter(document_type %in% c("Minutes", "Policy Statement")) %>%
    transmute(
      document_type = recode(
        document_type,
        "Policy Statement" = "Policy statements"
      ),
      sentence_count
    ),
  read_csv(press_document_level_path, show_col_types = FALSE) %>%
    transmute(
      document_type = "Press conferences",
      sentence_count
    )
) %>%
  mutate(
    document_type = factor(
      document_type,
      levels = c("Policy statements", "Minutes", "Press conferences")
    )
  )

agent_share_vars <- c(
  "agent_share_households",
  "agent_share_firms",
  "agent_share_financial sector",
  "agent_share_government",
  "agent_share_central bank"
)

agent_net_sentiment_vars <- c(
  "net_sentiment_households",
  "net_sentiment_firms",
  "net_sentiment_financial sector",
  "net_sentiment_government",
  "net_sentiment_central bank"
)

document_agent_shares <- bind_rows(
  read_csv(fomc_document_level_path, show_col_types = FALSE) %>%
    filter(document_type %in% c("Minutes", "Policy Statement")) %>%
    select(document_id, document_type, all_of(agent_share_vars)),
  read_csv(press_document_level_path, show_col_types = FALSE) %>%
    mutate(document_type = "Press conferences") %>%
    select(document_id, document_type, all_of(agent_share_vars))
) %>%
  mutate(
    document_type = recode(
      document_type,
      "Policy Statement" = "Policy statements"
    ),
    document_type = factor(
      document_type,
      levels = c("Policy statements", "Minutes", "Press conferences")
    )
  ) %>%
  pivot_longer(
    cols = all_of(agent_share_vars),
    names_to = "agent",
    values_to = "sentence_share"
  ) %>%
  mutate(
    agent = str_remove(agent, "^agent_share_"),
    agent = str_replace_all(agent, "_", " "),
    agent = str_to_title(agent),
    agent = factor(
      agent,
      levels = c(
        "Households",
        "Firms",
        "Financial Sector",
        "Government",
        "Central Bank"
      )
    )
  )

document_agent_net_sentiment <- bind_rows(
  read_csv(fomc_document_level_path, show_col_types = FALSE) %>%
    filter(document_type %in% c("Minutes", "Policy Statement")) %>%
    transmute(
      document_id,
      document_type,
      meeting_date = as.Date(parse_date_time(meeting_date, orders = c("ymd", "mdy"))),
      date = as.Date(parse_date_time(document_date, orders = c("ymd", "mdy"))),
      across(all_of(agent_net_sentiment_vars))
    ),
  read_csv(press_document_level_path, show_col_types = FALSE) %>%
    mutate(document_type = "Press conferences") %>%
    transmute(
      document_id,
      document_type,
      meeting_date = as.Date(parse_date_time(meeting_date, orders = c("ymd", "mdy"))),
      date = as.Date(parse_date_time(meeting_date, orders = c("ymd", "mdy"))),
      across(all_of(agent_net_sentiment_vars))
    )
) %>%
  mutate(
    document_type = recode(
      document_type,
      "Policy Statement" = "Policy statements"
    ),
    document_type = factor(
      document_type,
      levels = c("Policy statements", "Minutes", "Press conferences")
    )
  ) %>%
  pivot_longer(
    cols = all_of(agent_net_sentiment_vars),
    names_to = "agent",
    values_to = "net_sentiment"
  ) %>%
  mutate(
    agent = str_remove(agent, "^net_sentiment_"),
    agent = str_replace_all(agent, "_", " "),
    agent = str_to_title(agent),
    agent = factor(
      agent,
      levels = c(
        "Households",
        "Firms",
        "Financial Sector",
        "Government",
        "Central Bank"
      )
    )
  )

# Quick checks
cat("Minutes observations:", nrow(minutes), "\n")
cat("Press conference observations:", nrow(press), "\n")
cat("Combined observations:", nrow(df_all), "\n")
cat("Documents with sentence counts:", nrow(document_sentence_counts), "\n")

cat("\nMissing MPS values:\n")
cat("Minutes:", sum(is.na(minutes$mps)), "\n")
cat("Press conferences:", sum(is.na(press$mps)), "\n")


# -----------------------------
# 3. Variable lists
# -----------------------------

sentiment_vars <- c(
  "s_Households",
  "s_Firms",
  "s_Financial_Sector",
  "s_Government",
  "s_Central_Bank"
)

assets <- c("spx", "zt", "btc")
horizons <- c("r0", "r1", "r2", "r3", "r5", "r7", "r9")

return_vars <- expand_grid(asset = assets, horizon = horizons) %>%
  mutate(return_var = paste0(asset, "_", horizon)) %>%
  pull(return_var)


# -----------------------------
# 4. Descriptive statistics
# -----------------------------

sentiment_summary <- df_all %>%
  select(communication_type, all_of(sentiment_vars)) %>%
  pivot_longer(
    cols = all_of(sentiment_vars),
    names_to = "sentiment_category",
    values_to = "sentiment"
  ) %>%
  group_by(communication_type, sentiment_category) %>%
  summarise(
    n = sum(!is.na(sentiment)),
    mean = mean(sentiment, na.rm = TRUE),
    sd = sd(sentiment, na.rm = TRUE),
    min = min(sentiment, na.rm = TRUE),
    p25 = quantile(sentiment, 0.25, na.rm = TRUE),
    median = median(sentiment, na.rm = TRUE),
    p75 = quantile(sentiment, 0.75, na.rm = TRUE),
    max = max(sentiment, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(sentiment_summary, file.path(output_dir, "sentiment_summary.csv"))

return_summary <- df_all %>%
  select(communication_type, all_of(return_vars)) %>%
  pivot_longer(
    cols = all_of(return_vars),
    names_to = "return_variable",
    values_to = "return"
  ) %>%
  group_by(communication_type, return_variable) %>%
  summarise(
    n = sum(!is.na(return)),
    mean = mean(return, na.rm = TRUE),
    sd = sd(return, na.rm = TRUE),
    min = min(return, na.rm = TRUE),
    p25 = quantile(return, 0.25, na.rm = TRUE),
    median = median(return, na.rm = TRUE),
    p75 = quantile(return, 0.75, na.rm = TRUE),
    max = max(return, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(return_summary, file.path(output_dir, "return_summary.csv"))

agent_share_summary <- document_agent_shares %>%
  group_by(document_type, agent) %>%
  summarise(
    n_documents = sum(!is.na(sentence_share)),
    mean_sentence_share = mean(sentence_share, na.rm = TRUE),
    sd_sentence_share = sd(sentence_share, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(
  agent_share_summary,
  file.path(output_dir, "agent_share_summary.csv")
)

agent_net_sentiment_summary <- document_agent_net_sentiment %>%
  group_by(document_type, agent) %>%
  summarise(
    n_documents = sum(!is.na(net_sentiment)),
    mean_net_sentiment = mean(net_sentiment, na.rm = TRUE),
    median_net_sentiment = median(net_sentiment, na.rm = TRUE),
    sd_net_sentiment = sd(net_sentiment, na.rm = TRUE),
    p25_net_sentiment = quantile(net_sentiment, 0.25, na.rm = TRUE),
    p75_net_sentiment = quantile(net_sentiment, 0.75, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(
  agent_net_sentiment_summary,
  file.path(output_dir, "agent_net_sentiment_summary.csv")
)


# -----------------------------
# 5. Descriptive plots
# -----------------------------

sentiment_long <- df_all %>%
  select(Date, communication_type, all_of(sentiment_vars)) %>%
  pivot_longer(
    cols = all_of(sentiment_vars),
    names_to = "sentiment_category",
    values_to = "sentiment"
  )

p_sentiment_density <- ggplot(sentiment_long, aes(x = sentiment)) +
  geom_density(
    color = research_colors$primary,
    fill = research_colors$primary_light,
    linewidth = 0.5,
    alpha = 0.7,
    na.rm = TRUE
  ) +
  facet_grid(communication_type ~ sentiment_category) +
  labs(
    x = "Net sentiment",
    y = "Density"
  ) +
  theme_analysis()

save_plot(
  filename_base = "sentiment_density",
  plot = p_sentiment_density,
  width = 12,
  height = 7
)

p_sentiment_time <- ggplot(sentiment_long, aes(x = Date, y = sentiment)) +
  geom_line(color = research_colors$primary, linewidth = 0.35, na.rm = TRUE) +
  geom_point(color = research_colors$primary, size = 0.8, alpha = 0.8, na.rm = TRUE) +
  facet_grid(communication_type ~ sentiment_category) +
  labs(
    x = "Date",
    y = "Net sentiment"
  ) +
  theme_analysis()

save_plot(
  filename_base = "sentiment_time_series",
  plot = p_sentiment_time,
  width = 12,
  height = 7
)

p_sentence_count_distribution <- ggplot(
  document_sentence_counts,
  aes(x = sentence_count, fill = document_type, color = document_type)
) +
  geom_histogram(
    binwidth = 25,
    boundary = 0,
    position = "identity",
    alpha = 0.32,
    linewidth = 0.25,
    na.rm = TRUE
  ) +
  scale_fill_manual(values = research_colors$document_type) +
  scale_color_manual(values = research_colors$document_type) +
  scale_x_continuous(
    breaks = seq(
      0,
      max(document_sentence_counts$sentence_count, na.rm = TRUE),
      by = 100
    )
  ) +
  labs(
    x = "Sentence count",
    y = "Frequency",
    fill = "Document type",
    color = "Document type"
  ) +
  theme_analysis()

save_plot(
  filename_base = "sentence_count_distribution",
  plot = p_sentence_count_distribution,
  width = 10,
  height = 6
)

p_agent_share <- agent_share_summary %>%
  arrange(document_type, agent) %>%
  mutate(
    share_label = paste0(round(100 * mean_sentence_share), "%")
  ) %>%
  ggplot(aes(x = "", y = mean_sentence_share, fill = agent)) +
  geom_col(width = 1, color = "white", linewidth = 0.4) +
  coord_polar(theta = "y") +
  geom_text(
    aes(label = share_label),
    position = position_stack(vjust = 0.5),
    color = research_colors$text,
    size = 3.2,
    family = plot_font_family
  ) +
  facet_wrap(~ document_type, nrow = 1) +
  scale_fill_manual(values = research_colors$agent) +
  labs(
    x = NULL,
    y = NULL,
    fill = "Economic agent"
  ) +
  theme_void(base_family = plot_font_family) +
  theme(
    legend.position = "bottom",
    legend.title = element_text(color = research_colors$text),
    legend.text = element_text(color = research_colors$text),
    strip.text = element_text(color = research_colors$text)
  )

save_plot(
  filename_base = "agent_share_by_document_type",
  plot = p_agent_share,
  width = 11,
  height = 4.8
)

p_agent_share_minutes_press <- agent_share_summary %>%
  filter(document_type %in% c("Minutes", "Press conferences")) %>%
  arrange(document_type, agent) %>%
  mutate(
    share_label = paste0(round(100 * mean_sentence_share), "%")
  ) %>%
  ggplot(aes(x = "", y = mean_sentence_share, fill = agent)) +
  geom_col(width = 1, color = "white", linewidth = 0.4) +
  coord_polar(theta = "y") +
  geom_text(
    aes(label = share_label),
    position = position_stack(vjust = 0.5),
    color = research_colors$text,
    size = 3.2,
    family = plot_font_family
  ) +
  facet_wrap(~ document_type, nrow = 1) +
  scale_fill_manual(values = research_colors$agent) +
  labs(
    x = NULL,
    y = NULL,
    fill = "Economic agent"
  ) +
  theme_void(base_family = plot_font_family) +
  theme(
    legend.position = "bottom",
    legend.title = element_text(color = research_colors$text),
    legend.text = element_text(color = research_colors$text),
    strip.text = element_text(color = research_colors$text)
  )

save_plot(
  filename_base = "agent_share_minutes_press_conferences",
  plot = p_agent_share_minutes_press,
  width = 8,
  height = 4.8
)

p_agent_net_sentiment <- document_agent_net_sentiment %>%
  ggplot(aes(x = agent, y = net_sentiment, fill = agent)) +
  geom_hline(yintercept = 0, linewidth = 0.3, color = research_colors$reference) +
  geom_boxplot(
    width = 0.55,
    alpha = 0.9,
    outlier.shape = 21,
    outlier.size = 1.2,
    outlier.stroke = 0.2,
    color = research_colors$text,
    na.rm = TRUE
  ) +
  facet_wrap(~ document_type, nrow = 1) +
  scale_fill_manual(values = research_colors$agent) +
  scale_y_continuous(
    limits = c(-1, 1),
    breaks = seq(-1, 1, by = 0.5)
  ) +
  labs(
    x = NULL,
    y = "Net sentiment",
    fill = "Economic agent"
  ) +
  theme_analysis() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 30, hjust = 1),
    strip.text = element_text(color = research_colors$text)
  )

save_plot(
  filename_base = "agent_net_sentiment_by_document_type",
  plot = p_agent_net_sentiment,
  width = 12,
  height = 5
)

p_agent_net_sentiment_minutes_press <- document_agent_net_sentiment %>%
  filter(document_type %in% c("Minutes", "Press conferences")) %>%
  ggplot(aes(x = agent, y = net_sentiment, fill = agent)) +
  geom_hline(yintercept = 0, linewidth = 0.3, color = research_colors$reference) +
  geom_boxplot(
    width = 0.55,
    alpha = 0.9,
    outlier.shape = 21,
    outlier.size = 1.2,
    outlier.stroke = 0.2,
    color = research_colors$text,
    na.rm = TRUE
  ) +
  facet_wrap(~ document_type, nrow = 1) +
  scale_fill_manual(values = research_colors$agent) +
  scale_y_continuous(
    limits = c(-1, 1),
    breaks = seq(-1, 1, by = 0.5)
  ) +
  labs(
    x = NULL,
    y = "Net sentiment",
    fill = "Economic agent"
  ) +
  theme_analysis() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 30, hjust = 1),
    strip.text = element_text(color = research_colors$text)
  )

save_plot(
  filename_base = "agent_net_sentiment_minutes_press_conferences",
  plot = p_agent_net_sentiment_minutes_press,
  width = 9,
  height = 5
)

p_policy_statement_net_sentiment <- document_agent_net_sentiment %>%
  filter(document_type == "Policy statements") %>%
  ggplot(aes(x = agent, y = net_sentiment, fill = agent)) +
  geom_hline(yintercept = 0, linewidth = 0.3, color = research_colors$reference) +
  geom_boxplot(
    width = 0.55,
    alpha = 0.9,
    outlier.shape = 21,
    outlier.size = 1.2,
    outlier.stroke = 0.2,
    color = research_colors$text,
    na.rm = TRUE
  ) +
  scale_fill_manual(values = research_colors$agent) +
  scale_y_continuous(
    limits = c(-1, 1),
    breaks = seq(-1, 1, by = 0.5)
  ) +
  labs(
    x = NULL,
    y = "Net sentiment",
    fill = "Economic agent"
  ) +
  theme_analysis() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 30, hjust = 1)
  )

p_sentence_count_and_policy_net_sentiment <- (
  p_sentence_count_distribution +
    theme(
      legend.position = "bottom",
      legend.title = element_text(color = research_colors$text),
      legend.text = element_text(color = research_colors$text)
    )
) + p_policy_statement_net_sentiment +
  plot_layout(widths = c(1.15, 1))

save_plot(
  filename_base = "sentence_count_distribution_and_policy_statement_net_sentiment",
  plot = p_sentence_count_and_policy_net_sentiment,
  width = 14,
  height = 5.5
)

agent_net_sentiment_rolling <- document_agent_net_sentiment %>%
  arrange(document_type, agent, date, document_id) %>%
  group_by(document_type, agent) %>%
  mutate(
    net_sentiment_roll4 = zoo::rollapplyr(
      net_sentiment,
      width = 4,
      FUN = mean,
      fill = NA_real_,
      partial = FALSE,
      na.rm = TRUE
    )
  ) %>%
  ungroup()

minutes_rolling_for_correlation <- agent_net_sentiment_rolling %>%
  filter(
    document_type == "Minutes",
    !is.na(net_sentiment_roll4)
  ) %>%
  transmute(
    agent,
    minutes_meeting_date = meeting_date,
    minutes_net_sentiment_roll4 = net_sentiment_roll4
  )

press_rolling_for_correlation <- agent_net_sentiment_rolling %>%
  filter(
    document_type == "Press conferences",
    !is.na(net_sentiment_roll4)
  ) %>%
  transmute(
    agent,
    press_meeting_date = meeting_date,
    press_net_sentiment_roll4 = net_sentiment_roll4
  )

minutes_press_rolling_correlation_data <- inner_join(
  minutes_rolling_for_correlation,
  press_rolling_for_correlation,
  by = "agent",
  relationship = "many-to-many"
) %>%
  filter(
    press_meeting_date >= minutes_meeting_date,
    press_meeting_date <= minutes_meeting_date + days(1)
  )

minutes_press_rolling_correlation <- minutes_press_rolling_correlation_data %>%
  group_by(agent) %>%
  summarise(
    n_paired_meetings = n(),
    first_press_meeting_date = min(press_meeting_date),
    last_press_meeting_date = max(press_meeting_date),
    correlation = if_else(
      n_paired_meetings >= 2,
      cor(minutes_net_sentiment_roll4, press_net_sentiment_roll4),
      NA_real_
    ),
    p_value = if_else(
      n_paired_meetings >= 3,
      cor.test(minutes_net_sentiment_roll4, press_net_sentiment_roll4)$p.value,
      NA_real_
    ),
    .groups = "drop"
  ) %>%
  mutate(
    significance = case_when(
      is.na(p_value) ~ "",
      p_value < 0.01 ~ "***",
      p_value < 0.05 ~ "**",
      p_value < 0.10 ~ "*",
      TRUE ~ ""
    )
  )

write_csv(
  minutes_press_rolling_correlation,
  file.path(output_dir, "agent_net_sentiment_rolling_minutes_press_correlation.csv")
)

correlation_table_plot_data <- minutes_press_rolling_correlation %>%
  transmute(
    agent = as.character(agent),
    correlation = sprintf("%.3f%s", correlation, significance),
    p_value = if_else(is.na(p_value), "", sprintf("%.3f", p_value)),
    n_paired_meetings = as.character(n_paired_meetings),
    sample_period = paste(first_press_meeting_date, last_press_meeting_date, sep = " to ")
  ) %>%
  pivot_longer(
    cols = -agent,
    names_to = "statistic",
    values_to = "value"
  ) %>%
  mutate(
    statistic = recode(
      statistic,
      "correlation" = "Correlation",
      "p_value" = "p-value",
      "n_paired_meetings" = "N",
      "sample_period" = "Sample period"
    ),
    statistic = factor(
      statistic,
      levels = c("Correlation", "p-value", "N", "Sample period")
    ),
    agent = factor(
      agent,
      levels = levels(document_agent_net_sentiment$agent)
    )
  )

p_minutes_press_correlation_table <- ggplot(
  correlation_table_plot_data,
  aes(x = statistic, y = agent)
) +
  geom_tile(fill = "white", color = "#D9D9D9", linewidth = 0.3) +
  geom_text(aes(label = value), family = plot_font_family, size = 3.4) +
  labs(
    x = NULL,
    y = NULL,
    caption = "* p < 0.10, ** p < 0.05, *** p < 0.01"
  ) +
  theme_minimal(base_family = plot_font_family) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(color = research_colors$text, face = "bold"),
    axis.text.y = element_text(color = research_colors$text),
    plot.caption = element_text(hjust = 0, color = research_colors$text)
  )

save_plot(
  filename_base = "agent_net_sentiment_rolling_minutes_press_correlation",
  plot = p_minutes_press_correlation_table,
  width = 8,
  height = 3.2
)

policy_periods <- tibble(
  period = paste0("P", 1:8),
  start_date = as.Date(c(
    "2012-01-01",
    "2015-01-01",
    "2019-01-01",
    "2020-03-01",
    "2021-01-01",
    "2022-03-01",
    "2023-08-01",
    "2025-01-01"
  )),
  end_date = as.Date(c(
    "2014-12-31",
    "2018-12-31",
    "2020-02-29",
    "2020-12-31",
    "2022-02-28",
    "2023-07-31",
    "2024-12-31",
    "2025-12-31"
  )),
  short_label = c(
    "Post-crisis\naccommodation",
    "Policy\nnormalization",
    "Late-cycle\npivot",
    "COVID-19\nshock",
    "Recovery /\ninflation buildup",
    "Rapid\ntightening",
    "Restrictive /\ndisinflation",
    "Wait-and-see\npolicy"
  ),
  description = c(
    "Post-crisis accommodation and tapering",
    "Gradual policy normalization",
    "Late-cycle slowdown and policy pivot",
    "COVID-19 shock and emergency stabilization",
    "Recovery, supply constraints, and inflation buildup",
    "Rapid tightening and inflation control",
    "Restrictive policy and cautious disinflation",
    "Wait-and-see policy under soft-landing conditions"
  )
) %>%
  mutate(
    midpoint = start_date + floor(as.numeric(end_date - start_date) / 2),
    year_label = paste0(year(start_date), "-", year(end_date)),
    period_fill = if_else(row_number() %% 2 == 1, "#F4FAFE", "#EAF4FB")
  )

write_csv(
  policy_periods,
  file.path(output_dir, "policy_period_definitions.csv")
)

sentence_count_distribution_data <- document_sentence_counts %>%
  filter(!is.na(sentence_count)) %>%
  mutate(
    bin_width = 25,
    bin_start = floor(sentence_count / bin_width) * bin_width,
    bin_end = bin_start + bin_width,
    bin_midpoint = bin_start + bin_width / 2
  ) %>%
  count(document_type, bin_width, bin_start, bin_end, bin_midpoint, name = "frequency")

plot_data_machine_readable <- bind_rows(
  sentence_count_distribution_data %>%
    transmute(
      chart = "sentence_count_distribution",
      document_type,
      agent = NA_character_,
      document_id = NA_character_,
      date = as.Date(NA),
      sentence_count = NA_real_,
      bin_width,
      bin_start,
      bin_end,
      bin_midpoint,
      frequency,
      mean_sentence_share = NA_real_,
      net_sentiment = NA_real_,
      net_sentiment_roll4 = NA_real_
    ),
  agent_share_summary %>%
    transmute(
      chart = "agent_share_by_document_type",
      document_type,
      agent = as.character(agent),
      document_id = NA_character_,
      date = as.Date(NA),
      sentence_count = NA_real_,
      bin_width = NA_real_,
      bin_start = NA_real_,
      bin_end = NA_real_,
      bin_midpoint = NA_real_,
      frequency = NA_integer_,
      mean_sentence_share,
      net_sentiment = NA_real_,
      net_sentiment_roll4 = NA_real_
    ),
  document_agent_net_sentiment %>%
    filter(!is.na(net_sentiment)) %>%
    transmute(
      chart = "agent_net_sentiment_by_document_type",
      document_type,
      agent = as.character(agent),
      document_id,
      date,
      sentence_count = NA_real_,
      bin_width = NA_real_,
      bin_start = NA_real_,
      bin_end = NA_real_,
      bin_midpoint = NA_real_,
      frequency = NA_integer_,
      mean_sentence_share = NA_real_,
      net_sentiment,
      net_sentiment_roll4 = NA_real_
    ),
  agent_net_sentiment_rolling %>%
    filter(!is.na(net_sentiment_roll4)) %>%
    left_join(
      policy_periods %>%
        select(period, short_label, description, start_date, end_date),
      by = join_by(date >= start_date, date <= end_date)
    ) %>%
    transmute(
      chart = "agent_net_sentiment_rolling_mean_by_document_type",
      document_type,
      agent = as.character(agent),
      document_id,
      date,
      period,
      period_label = short_label,
      period_description = description,
      sentence_count = NA_real_,
      bin_width = NA_real_,
      bin_start = NA_real_,
      bin_end = NA_real_,
      bin_midpoint = NA_real_,
      frequency = NA_integer_,
      mean_sentence_share = NA_real_,
      net_sentiment,
      net_sentiment_roll4
    )
)

write_csv(
  plot_data_machine_readable,
  file.path(output_dir, "plot_data_machine_readable.csv")
)

p_agent_net_sentiment_rolling <- agent_net_sentiment_rolling %>%
  filter(!is.na(net_sentiment_roll4)) %>%
  ggplot(aes(x = date, y = net_sentiment_roll4, color = agent)) +
  geom_rect(
    data = policy_periods,
    aes(
      xmin = start_date,
      xmax = end_date,
      ymin = -Inf,
      ymax = Inf,
      fill = period_fill
    ),
    inherit.aes = FALSE,
    alpha = 0.7
  ) +
  geom_vline(
    data = policy_periods %>% filter(period != "P1"),
    aes(xintercept = start_date),
    color = "#B7D7EE",
    linewidth = 0.25
  ) +
  geom_text(
    data = policy_periods,
    aes(
      x = midpoint,
      y = 0.98,
      label = paste0(period, "\n", year_label)
    ),
    inherit.aes = FALSE,
    color = research_colors$text,
    size = 2.7,
    lineheight = 0.85,
    family = plot_font_family
  ) +
  geom_hline(yintercept = 0, linewidth = 0.3, color = research_colors$reference) +
  geom_line(linewidth = 0.55, na.rm = TRUE) +
  geom_point(size = 0.8, alpha = 0.8, na.rm = TRUE) +
  facet_wrap(~ document_type, ncol = 1) +
  scale_fill_identity() +
  scale_color_manual(values = research_colors$agent) +
  scale_x_date(
    limits = c(min(policy_periods$start_date), max(policy_periods$end_date)),
    breaks = policy_periods$midpoint,
    labels = policy_periods$short_label,
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    limits = c(-1, 1),
    breaks = seq(-1, 1, by = 0.5)
  ) +
  labs(
    x = "Policy period",
    y = "Net sentiment, 4-document rolling mean",
    color = "Economic agent"
  ) +
  theme_analysis() +
  theme(
    legend.position = "bottom",
    legend.title = element_text(color = research_colors$text),
    legend.text = element_text(color = research_colors$text),
    axis.text.x = element_text(size = 7.2, lineheight = 0.88),
    strip.text = element_text(color = research_colors$text)
  )

save_plot(
  filename_base = "agent_net_sentiment_rolling_mean_by_document_type",
  plot = p_agent_net_sentiment_rolling,
  width = 14,
  height = 9
)

p_agent_net_sentiment_rolling_minutes_press <- agent_net_sentiment_rolling %>%
  filter(
    document_type %in% c("Minutes", "Press conferences"),
    !is.na(net_sentiment_roll4)
  ) %>%
  ggplot(aes(x = date, y = net_sentiment_roll4, color = agent)) +
  geom_rect(
    data = policy_periods,
    aes(
      xmin = start_date,
      xmax = end_date,
      ymin = -Inf,
      ymax = Inf,
      fill = period_fill
    ),
    inherit.aes = FALSE,
    alpha = 0.7
  ) +
  geom_vline(
    data = policy_periods %>% filter(period != "P1"),
    aes(xintercept = start_date),
    color = "#B7D7EE",
    linewidth = 0.25
  ) +
  geom_text(
    data = policy_periods,
    aes(
      x = midpoint,
      y = 0.98,
      label = paste0(period, "\n", year_label)
    ),
    inherit.aes = FALSE,
    color = research_colors$text,
    size = 2.7,
    lineheight = 0.85,
    family = plot_font_family
  ) +
  geom_hline(yintercept = 0, linewidth = 0.3, color = research_colors$reference) +
  geom_line(linewidth = 0.55, na.rm = TRUE) +
  geom_point(size = 0.8, alpha = 0.8, na.rm = TRUE) +
  facet_wrap(~ document_type, ncol = 1) +
  scale_fill_identity() +
  scale_color_manual(values = research_colors$agent) +
  scale_x_date(
    limits = c(min(policy_periods$start_date), max(policy_periods$end_date)),
    breaks = policy_periods$midpoint,
    labels = policy_periods$short_label,
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    limits = c(-1, 1),
    breaks = seq(-1, 1, by = 0.5)
  ) +
  labs(
    x = "Policy period",
    y = "Net sentiment, 4-document rolling mean",
    color = "Economic agent"
  ) +
  theme_analysis() +
  theme(
    legend.position = "bottom",
    legend.title = element_text(color = research_colors$text),
    legend.text = element_text(color = research_colors$text),
    axis.text.x = element_text(size = 7.2, lineheight = 0.88),
    strip.text = element_text(color = research_colors$text)
  )

save_plot(
  filename_base = "agent_net_sentiment_rolling_mean_minutes_press_conferences",
  plot = p_agent_net_sentiment_rolling_minutes_press,
  width = 14,
  height = 7
)


# -----------------------------
# 6. Helper functions for regressions
# -----------------------------

robust_lm_tidy <- function(model) {
  robust_vcov <- sandwich::vcovHC(model, type = "HC1")
  lmtest::coeftest(model, vcov. = robust_vcov) %>%
    broom::tidy() %>%
    mutate(
      r.squared = summary(model)$r.squared,
      adj.r.squared = summary(model)$adj.r.squared,
      nobs = nobs(model)
    )
}

run_sentiment_regression <- function(data, return_var, sentiment_var, use_mps = FALSE) {
  if (use_mps) {
    reg_data <- data %>%
      select(all_of(c(return_var, sentiment_var, "mps"))) %>%
      drop_na()
    
    formula <- as.formula(paste(return_var, "~", sentiment_var, "+ mps"))
  } else {
    reg_data <- data %>%
      select(all_of(c(return_var, sentiment_var))) %>%
      drop_na()
    
    formula <- as.formula(paste(return_var, "~", sentiment_var))
  }
  
  if (nrow(reg_data) < 10) {
    return(tibble())
  }
  
  model <- lm(formula, data = reg_data)
  
  robust_lm_tidy(model) %>%
    mutate(
      return_var = return_var,
      sentiment_var = sentiment_var,
      use_mps = use_mps
    )
}

run_grid <- function(data, communication_type_label, asset, sentiment_set, use_mps = FALSE) {
  expand_grid(
    horizon = horizons,
    sentiment_var = sentiment_set
  ) %>%
    mutate(
      return_var = paste0(asset, "_", horizon)
    ) %>%
    pmap_dfr(function(horizon, sentiment_var, return_var) {
      run_sentiment_regression(
        data = data,
        return_var = return_var,
        sentiment_var = sentiment_var,
        use_mps = use_mps
      ) %>%
        mutate(
          communication_type = communication_type_label,
          asset = asset,
          horizon = horizon
        )
    })
}

extract_sentiment_coefficients <- function(results) {
  results %>%
    filter(term == sentiment_var) %>%
    transmute(
      hypothesis,
      communication_type,
      sample,
      asset,
      horizon,
      sentiment_var,
      estimate,
      std_error = std.error,
      statistic,
      p_value = p.value,
      r_squared = r.squared,
      adj_r_squared = adj.r.squared,
      nobs,
      use_mps
    )
}

extract_sentiment_coefficients_with_period <- function(results) {
  results %>%
    filter(term == sentiment_var) %>%
    transmute(
      hypothesis,
      communication_type,
      sample,
      asset,
      horizon,
      sentiment_var,
      estimate,
      std_error = std.error,
      statistic,
      p_value = p.value,
      r_squared = r.squared,
      adj_r_squared = adj.r.squared,
      nobs,
      use_mps
    )
}


# -----------------------------
# 7. Regression samples
# -----------------------------

minutes_full <- minutes

press_full <- press

press_mps_available <- press %>%
  filter(!is.na(mps))

cat("\nPress conference observations with non-missing MPS:", nrow(press_mps_available), "\n")


# -----------------------------
# 8. H1 regressions
# H1: Household and firm sentiment should be positively associated
#     with S&P 500 returns.
# -----------------------------

h1_sentiments <- c("s_Households", "s_Firms")

h1_minutes <- run_grid(
  data = minutes_full,
  communication_type_label = "Minutes",
  asset = "spx",
  sentiment_set = h1_sentiments,
  use_mps = FALSE
) %>%
  mutate(
    hypothesis = "H1",
    sample = "Full sample"
  )

h1_press_full_no_mps <- run_grid(
  data = press_full,
  communication_type_label = "Press conferences",
  asset = "spx",
  sentiment_set = h1_sentiments,
  use_mps = FALSE
) %>%
  mutate(
    hypothesis = "H1",
    sample = "Full sample, no MPS"
  )

h1_press_restricted_no_mps <- run_grid(
  data = press_mps_available,
  communication_type_label = "Press conferences",
  asset = "spx",
  sentiment_set = h1_sentiments,
  use_mps = FALSE
) %>%
  mutate(
    hypothesis = "H1",
    sample = "MPS-available sample, no MPS"
  )

h1_press_with_mps <- run_grid(
  data = press_mps_available,
  communication_type_label = "Press conferences",
  asset = "spx",
  sentiment_set = h1_sentiments,
  use_mps = TRUE
) %>%
  mutate(
    hypothesis = "H1",
    sample = "MPS-available sample, with MPS"
  )


# -----------------------------
# 9. H2 regressions
# H2: Household and firm sentiment should be negatively
#     associated with two-year Treasury futures returns.
# -----------------------------

h2_sentiments <- c("s_Households", "s_Firms")

h2_minutes <- run_grid(
  data = minutes_full,
  communication_type_label = "Minutes",
  asset = "zt",
  sentiment_set = h2_sentiments,
  use_mps = FALSE
) %>%
  mutate(
    hypothesis = "H2",
    sample = "Full sample"
  )

h2_press_full_no_mps <- run_grid(
  data = press_full,
  communication_type_label = "Press conferences",
  asset = "zt",
  sentiment_set = h2_sentiments,
  use_mps = FALSE
) %>%
  mutate(
    hypothesis = "H2",
    sample = "Full sample, no MPS"
  )

h2_press_restricted_no_mps <- run_grid(
  data = press_mps_available,
  communication_type_label = "Press conferences",
  asset = "zt",
  sentiment_set = h2_sentiments,
  use_mps = FALSE
) %>%
  mutate(
    hypothesis = "H2",
    sample = "MPS-available sample, no MPS"
  )

h2_press_with_mps <- run_grid(
  data = press_mps_available,
  communication_type_label = "Press conferences",
  asset = "zt",
  sentiment_set = h2_sentiments,
  use_mps = TRUE
) %>%
  mutate(
    hypothesis = "H2",
    sample = "MPS-available sample, with MPS"
  )


# -----------------------------
# 10. H3 regressions
# H3: Financial-sector sentiment should have stronger explanatory power
#     for SPX and BTC than for ZT.
# -----------------------------

h3_sentiments <- c("s_Financial_Sector")

h3_minutes <- map_dfr(assets, function(asset_name) {
  run_grid(
    data = minutes_full,
    communication_type_label = "Minutes",
    asset = asset_name,
    sentiment_set = h3_sentiments,
    use_mps = FALSE
  )
}) %>%
  mutate(
    hypothesis = "H3",
    sample = "Full sample"
  )

h3_press_full_no_mps <- map_dfr(assets, function(asset_name) {
  run_grid(
    data = press_full,
    communication_type_label = "Press conferences",
    asset = asset_name,
    sentiment_set = h3_sentiments,
    use_mps = FALSE
  )
}) %>%
  mutate(
    hypothesis = "H3",
    sample = "Full sample, no MPS"
  )

h3_press_restricted_no_mps <- map_dfr(assets, function(asset_name) {
  run_grid(
    data = press_mps_available,
    communication_type_label = "Press conferences",
    asset = asset_name,
    sentiment_set = h3_sentiments,
    use_mps = FALSE
  )
}) %>%
  mutate(
    hypothesis = "H3",
    sample = "MPS-available sample, no MPS"
  )

h3_press_with_mps <- map_dfr(assets, function(asset_name) {
  run_grid(
    data = press_mps_available,
    communication_type_label = "Press conferences",
    asset = asset_name,
    sentiment_set = h3_sentiments,
    use_mps = TRUE
  )
}) %>%
  mutate(
    hypothesis = "H3",
    sample = "MPS-available sample, with MPS"
  )


# -----------------------------
# 10b. H3 Bitcoin robustness regressions
# Bitcoin only. The robustness windows mirror the original H3 sample structure:
# 2020-2025 replaces the full sample, and 2020-2023 replaces the
# MPS-available sample.
# -----------------------------

h3_btc_robustness_minutes <- minutes_full %>%
  filter(Date >= as.Date("2020-01-01"), Date <= as.Date("2025-12-31")) %>%
  run_grid(
    communication_type_label = "Minutes",
    asset = "btc",
    sentiment_set = h3_sentiments,
    use_mps = FALSE
  ) %>%
  mutate(
    hypothesis = "H3 Bitcoin robustness",
    sample = "2020-2025"
  )

h3_btc_robustness_press_full_no_mps <- press_full %>%
  filter(Date >= as.Date("2020-01-01"), Date <= as.Date("2025-12-31")) %>%
  run_grid(
    communication_type_label = "Press conferences",
    asset = "btc",
    sentiment_set = h3_sentiments,
    use_mps = FALSE
  ) %>%
  mutate(
    hypothesis = "H3 Bitcoin robustness",
    sample = "2020-2025, no MPS"
  )

h3_btc_robustness_press_restricted_no_mps <- press_mps_available %>%
  filter(Date >= as.Date("2020-01-01"), Date <= as.Date("2023-12-31")) %>%
  run_grid(
    communication_type_label = "Press conferences",
    asset = "btc",
    sentiment_set = h3_sentiments,
    use_mps = FALSE
  ) %>%
  mutate(
    hypothesis = "H3 Bitcoin robustness",
    sample = "2020-2023, no MPS"
  )

h3_btc_robustness_press_with_mps <- press_mps_available %>%
  filter(Date >= as.Date("2020-01-01"), Date <= as.Date("2023-12-31")) %>%
  run_grid(
    communication_type_label = "Press conferences",
    asset = "btc",
    sentiment_set = h3_sentiments,
    use_mps = TRUE
  ) %>%
  mutate(
    hypothesis = "H3 Bitcoin robustness",
    sample = "2020-2023, with MPS"
  )


# -----------------------------
# 11. Combine and save regression output
# -----------------------------

all_results_raw <- bind_rows(
  h1_minutes,
  h1_press_full_no_mps,
  h1_press_restricted_no_mps,
  h1_press_with_mps,
  h2_minutes,
  h2_press_full_no_mps,
  h2_press_restricted_no_mps,
  h2_press_with_mps,
  h3_minutes,
  h3_press_full_no_mps,
  h3_press_restricted_no_mps,
  h3_press_with_mps
)

h3_btc_robustness_raw <- bind_rows(
  h3_btc_robustness_minutes,
  h3_btc_robustness_press_full_no_mps,
  h3_btc_robustness_press_restricted_no_mps,
  h3_btc_robustness_press_with_mps
) %>%
  mutate(
    sample = factor(
      sample,
      levels = c(
        "2020-2025",
        "2020-2025, no MPS",
        "2020-2023, no MPS",
        "2020-2023, with MPS"
      )
    )
  )

write_csv(
  all_results_raw,
  file.path(output_dir, "all_regression_results_raw.csv")
)

h3_btc_robustness_coefficients <- extract_sentiment_coefficients_with_period(h3_btc_robustness_raw)

write_csv(
  h3_btc_robustness_raw,
  file.path(output_dir, "H3_btc_robustness_regression_results_raw.csv")
)

sentiment_coefficients <- extract_sentiment_coefficients(all_results_raw)

write_csv(
  sentiment_coefficients,
  file.path(output_dir, "sentiment_coefficients.csv")
)

cat("\nSaved regression results to:", output_dir, "\n")


# -----------------------------
# 12. Create compact tables by hypothesis
# -----------------------------

h1_table <- sentiment_coefficients %>%
  filter(hypothesis == "H1") %>%
  arrange(communication_type, sample, sentiment_var, horizon)

h2_table <- sentiment_coefficients %>%
  filter(hypothesis == "H2") %>%
  arrange(communication_type, sample, sentiment_var, horizon)

h3_table <- sentiment_coefficients %>%
  filter(hypothesis == "H3") %>%
  arrange(communication_type, sample, asset, horizon)

h3_btc_robustness_table <- h3_btc_robustness_coefficients %>%
  arrange(communication_type, sample, asset, horizon)

write_csv(h1_table, file.path(output_dir, "H1_sentiment_coefficients.csv"))
write_csv(h2_table, file.path(output_dir, "H2_sentiment_coefficients.csv"))
write_csv(h3_table, file.path(output_dir, "H3_sentiment_coefficients.csv"))
write_csv(h3_btc_robustness_table, file.path(output_dir, "H3_btc_robustness_sentiment_coefficients.csv"))


# -----------------------------
# 13. Plot coefficient paths by horizon
# -----------------------------

plot_coefficients <- function(data, file_name) {
  p <- data %>%
    mutate(
      horizon_numeric = readr::parse_number(horizon),
      lower_95 = estimate - 1.96 * std_error,
      upper_95 = estimate + 1.96 * std_error
    ) %>%
    ggplot(aes(x = horizon_numeric, y = estimate)) +
    geom_hline(yintercept = 0, linewidth = 0.3, color = research_colors$reference) +
    geom_ribbon(
      aes(ymin = lower_95, ymax = upper_95),
      fill = research_colors$primary_light,
      alpha = 0.65
    ) +
    geom_line(color = research_colors$primary, linewidth = 0.45) +
    geom_point(color = research_colors$primary, size = 1.2) +
    facet_grid(communication_type + sample ~ sentiment_var + asset, scales = "free_y") +
    labs(
      x = "Calendar-day horizon",
      y = "Sentiment coefficient",
      caption = "Shaded bands show robust 95% confidence intervals based on coefficient +/- 1.96 x robust SE."
    ) +
    theme_analysis() +
    theme(
      plot.caption = element_text(hjust = 0, color = research_colors$text)
    )
  
  save_plot(
    filename_base = file_name,
    plot = p,
    width = 14,
    height = 9
  )
}

plot_coefficients(
  data = h1_table,
  file_name = "H1_coefficients"
)

plot_coefficients(
  data = h2_table,
  file_name = "H2_coefficients"
)

plot_coefficients(
  data = h3_table,
  file_name = "H3_coefficients"
)

plot_coefficients(
  data = h3_btc_robustness_table,
  file_name = "H3_btc_robustness_coefficients"
)


# -----------------------------
# 14. Optional: correlation matrices of sentiment variables
# -----------------------------

sentiment_correlation_minutes <- minutes %>%
  select(all_of(sentiment_vars)) %>%
  cor(use = "pairwise.complete.obs")

sentiment_correlation_press <- press %>%
  select(all_of(sentiment_vars)) %>%
  cor(use = "pairwise.complete.obs")

write_csv(
  as.data.frame(sentiment_correlation_minutes) %>%
    rownames_to_column("variable"),
  file.path(output_dir, "sentiment_correlation_minutes.csv")
)

write_csv(
  as.data.frame(sentiment_correlation_press) %>%
    rownames_to_column("variable"),
  file.path(output_dir, "sentiment_correlation_press.csv")
)


# -----------------------------
# 15. Console preview
# -----------------------------

cat("\nH1 preview:\n")
print(h1_table %>% select(communication_type, sample, horizon, sentiment_var, estimate, std_error, p_value, nobs) %>% head(20))

cat("\nH2 preview:\n")
print(h2_table %>% select(communication_type, sample, horizon, sentiment_var, estimate, std_error, p_value, nobs) %>% head(20))

cat("\nH3 preview:\n")
print(h3_table %>% select(communication_type, sample, asset, horizon, estimate, std_error, p_value, r_squared, nobs) %>% head(20))

cat("\nH3 Bitcoin robustness preview:\n")
print(h3_btc_robustness_table %>% select(communication_type, sample, asset, horizon, estimate, std_error, p_value, r_squared, nobs) %>% head(20))
