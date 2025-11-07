use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    env,
    f64::consts::TAU,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use color_eyre::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    DefaultTerminal, Frame,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{
        Axis, Bar, BarChart, BarGroup, Block, Cell, Chart, Dataset, GraphType, Paragraph, Row,
        Scrollbar, ScrollbarOrientation, ScrollbarState, Table, TableState, Wrap,
        canvas::{Canvas, Context as CanvasContext, Painter, Shape},
    },
};
use regex::Regex;

fn main() -> Result<()> {
    color_eyre::install()?;
    let terminal = ratatui::init();
    let csv_path = env::args().nth(1).map(PathBuf::from);
    let app = App::new(csv_path)?;
    let result = app.run(terminal);
    ratatui::restore();
    result
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChartMode {
    Line,
    Histogram,
    Pie,
}

impl ChartMode {
    fn cycle(self) -> Self {
        match self {
            Self::Line => Self::Histogram,
            Self::Histogram => Self::Pie,
            Self::Pie => Self::Line,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Line => "Line",
            Self::Histogram => "Histogram",
            Self::Pie => "Pie",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchScope {
    Global,
    Column(usize),
}

impl SearchScope {
    fn label(self, headers: &[String]) -> String {
        match self {
            SearchScope::Global => "Global".to_string(),
            SearchScope::Column(idx) => headers
                .get(idx)
                .cloned()
                .unwrap_or_else(|| format!("Column {}", idx.saturating_add(1))),
        }
    }
}

#[derive(Debug, Clone)]
enum InputMode {
    Normal,
    Searching(SearchInput),
}

#[derive(Debug, Clone)]
struct SearchInput {
    scope: SearchScope,
    buffer: String,
}

impl SearchInput {
    fn new(scope: SearchScope) -> Self {
        Self {
            scope,
            buffer: String::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CellPosition {
    row: usize,
    col: usize,
}

#[derive(Debug, Clone)]
struct SearchResults {
    scope: SearchScope,
    pattern: String,
    matches: Vec<CellPosition>,
    match_lookup: HashSet<CellPosition>,
    current_index: usize,
}

impl SearchResults {
    fn new(scope: SearchScope, pattern: String, matches: Vec<CellPosition>) -> Self {
        let match_lookup = matches.iter().copied().collect();
        Self {
            scope,
            pattern,
            matches,
            match_lookup,
            current_index: 0,
        }
    }

    fn is_match(&self, row: usize, col: usize) -> bool {
        self.match_lookup.contains(&CellPosition { row, col })
    }

    fn len(&self) -> usize {
        self.matches.len()
    }
}

#[derive(Debug)]
struct App {
    running: bool,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    original_rows: Vec<Vec<String>>,
    csv_path: Option<PathBuf>,
    selected_row: usize,
    selected_col: usize,
    selection_active: bool,
    column_offset: usize,
    sort_column: Option<usize>,
    sort_ascending: bool,
    chart_mode: ChartMode,
    status_message: String,
    table_state: TableState,
    input_mode: InputMode,
    search_results: Option<SearchResults>,
    search_cursor_visible: bool,
    last_cursor_toggle: Instant,
}

impl App {
    fn new(csv_path: Option<PathBuf>) -> Result<Self> {
        let mut app = Self {
            running: false,
            headers: Vec::new(),
            rows: Vec::new(),
            original_rows: Vec::new(),
            csv_path: None,
            selected_row: 0,
            selected_col: 0,
            selection_active: true,
            column_offset: 0,
            sort_column: None,
            sort_ascending: true,
            chart_mode: ChartMode::Line,
            status_message: "Ready".to_string(),
            table_state: TableState::default(),
            input_mode: InputMode::Normal,
            search_results: None,
            search_cursor_visible: true,
            last_cursor_toggle: Instant::now(),
        };

        if let Some(path) = csv_path {
            match Self::load_csv(&path) {
                Ok((headers, rows)) => {
                    app.headers = headers;
                    app.rows = rows;
                    app.original_rows = app.rows.clone();
                    app.csv_path = Some(path);
                    app.ensure_selection_in_bounds();
                    app.set_status(format!(
                        "Loaded {} rows × {} columns",
                        app.rows.len(),
                        app.headers.len()
                    ));
                }
                Err(error) => {
                    app.set_status(format!("Failed to load CSV: {error}"));
                }
            }
        } else {
            app.set_status("No CSV path supplied. Run: cargo run -- <file.csv>");
        }

        Ok(app)
    }

    fn run(mut self, mut terminal: DefaultTerminal) -> Result<()> {
        const TICK_RATE: Duration = Duration::from_millis(100);
        let mut last_tick = Instant::now();

        self.running = true;
        while self.running {
            terminal.draw(|frame| self.render(frame))?;

            let timeout = TICK_RATE
                .checked_sub(last_tick.elapsed())
                .unwrap_or(Duration::from_millis(0));

            if event::poll(timeout)? {
                self.handle_crossterm_events()?;
            }

            if last_tick.elapsed() >= TICK_RATE {
                self.on_tick();
                last_tick = Instant::now();
            }
        }
        Ok(())
    }

    fn render(&mut self, frame: &mut Frame) {
        self.ensure_selection_in_bounds();

        let area = frame.area();
        if area.height < 5 || area.width < 20 {
            frame.render_widget(
                Paragraph::new("Terminal too small for CSV viewer.")
                    .alignment(Alignment::Center)
                    .block(Block::bordered().title("Ratatui CSV Viewer")),
                area,
            );
            return;
        }

        let layout = Layout::vertical([
            Constraint::Length(5),
            Constraint::Percentage(55),
            Constraint::Percentage(40),
        ]);
        let chunks = layout.split(area);

        self.render_header(frame, chunks[0]);
        self.render_table(frame, chunks[1]);
        self.render_chart(frame, chunks[2]);
    }

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let file_line = match &self.csv_path {
            Some(path) => format!("File: {}", path.display()),
            None => "File: <not loaded>".to_string(),
        };

        let selected_label = if self.selection_active {
            format!(
                "row {} / col {}",
                self.selected_row,
                self.selected_col.saturating_add(1)
            )
        } else {
            "none".to_string()
        };

        let mut first_line = vec![Span::raw(file_line), Span::raw("  ")];
        first_line.extend(self.search_spans());

        let mut lines = vec![Line::from(first_line)];
        lines.push(Line::from(format!(
            "Rows: {}  Columns: {}  Selected: {}",
            self.rows.len(),
            self.headers.len(),
            selected_label,
        )));
        lines.push(Line::from(format!(
            "Sort: Enter on header (press 'r' to reset) | Charts: Tab to cycle (mode {}) or press 1/2/3",
            self.chart_mode.label()
        )));
        lines.push(Line::from(vec![
            Span::raw("Status: "),
            Span::styled(
                &self.status_message,
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));

        lines.push(Line::from(
            "Navigation: Arrows move, Home/End jump, PgUp/PgDn faster, '/' starts regex search (global or column), Enter cycles matches, Esc clears selection then quits, q quits",
        ));

        frame.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: true })
                .block(Block::bordered().title("CSV Viewer")),
            area,
        );
    }

    fn on_tick(&mut self) {
        self.update_search_cursor();
    }

    fn update_search_cursor(&mut self) {
        const BLINK_INTERVAL: Duration = Duration::from_millis(500);
        if matches!(self.input_mode, InputMode::Searching(_)) {
            let now = Instant::now();
            if now.duration_since(self.last_cursor_toggle) >= BLINK_INTERVAL {
                self.search_cursor_visible = !self.search_cursor_visible;
                self.last_cursor_toggle = now;
            }
        } else {
            self.search_cursor_visible = true;
            self.last_cursor_toggle = Instant::now();
        }
    }

    fn search_spans(&self) -> Vec<Span<'static>> {
        let prompt_style = Style::default().fg(Color::Cyan);
        let input_style = Style::default().fg(Color::Green);
        let accent_style = Style::default().fg(Color::Rgb(255, 165, 0));

        match &self.input_mode {
            InputMode::Searching(input) => {
                let scope = input.scope.label(&self.headers);
                let cursor = if self.search_cursor_visible { "|" } else { " " };
                vec![
                    Span::styled(format!("Search [{}] regex: /", scope), prompt_style),
                    Span::styled(format!("{}{}", input.buffer, cursor), input_style),
                ]
            }
            InputMode::Normal => {
                if let Some(results) = &self.search_results {
                    let scope = results.scope.label(&self.headers);
                    if results.len() > 0 {
                        vec![
                            Span::styled(format!("Search [{}]: /", scope), prompt_style),
                            Span::styled(results.pattern.clone(), input_style),
                            Span::raw("  "),
                            Span::styled(
                                format!("match {}/{}", results.current_index + 1, results.len()),
                                accent_style,
                            ),
                        ]
                    } else {
                        vec![
                            Span::styled(format!("Search [{}]: /", scope), prompt_style),
                            Span::styled(results.pattern.clone(), input_style),
                            Span::raw("  "),
                            Span::styled("no matches".to_string(), accent_style),
                        ]
                    }
                } else {
                    vec![Span::styled(
                        "Search: press '/' to start regex search".to_string(),
                        prompt_style,
                    )]
                }
            }
        }
    }

    fn render_table(&mut self, frame: &mut Frame, area: Rect) {
        if self.headers.is_empty() {
            frame.render_widget(
                Paragraph::new("No table data. Provide a CSV file.")
                    .alignment(Alignment::Center)
                    .block(Block::bordered().title("Table")),
                area,
            );
            return;
        }

        let column_widths = self.compute_column_widths();
        self.adjust_column_offset(area.width, &column_widths);
        let visible_columns = self.visible_columns(area.width, &column_widths);
        if visible_columns.is_empty() {
            frame.render_widget(
                Paragraph::new("No columns fit in the available space.")
                    .alignment(Alignment::Center)
                    .block(Block::bordered().title("Table")),
                area,
            );
            return;
        }

        let widths = visible_columns
            .iter()
            .map(|&idx| Constraint::Length(column_widths[idx]))
            .collect::<Vec<_>>();

        let header_row = {
            let cells = visible_columns
                .iter()
                .map(|&idx| {
                    let header = self
                        .headers
                        .get(idx)
                        .map(String::as_str)
                        .unwrap_or_default();
                    let indicator = match (self.sort_column, self.sort_ascending) {
                        (Some(col), true) if col == idx => " ^",
                        (Some(col), false) if col == idx => " v",
                        _ => "",
                    };
                    let mut style = Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD);
                    if self.selected_row == 0 && self.selected_col == idx {
                        style = style
                            .bg(Color::Yellow)
                            .fg(Color::Black)
                            .add_modifier(Modifier::BOLD);
                    }
                    Cell::from(format!("{header}{indicator}")).style(style)
                })
                .collect::<Vec<_>>();
            Row::new(cells)
        };

        let rows = self
            .rows
            .iter()
            .enumerate()
            .map(|(row_idx, data_row)| {
                let row_cells = visible_columns
                    .iter()
                    .map(|&col_idx| {
                        let content = data_row
                            .get(col_idx)
                            .map(String::as_str)
                            .unwrap_or_default();

                        let mut style = Style::default();
                        let is_selected_cell = self.selection_active
                            && self.selected_row == row_idx + 1
                            && self.selected_col == col_idx;
                        let is_selected_row = self.selection_active
                            && self.selected_row == row_idx + 1
                            && !is_selected_cell;
                        let is_selected_col = self.selection_active
                            && self.selected_col == col_idx
                            && !is_selected_cell;

                        if is_selected_cell {
                            style = style
                                .bg(Color::Cyan)
                                .fg(Color::Black)
                                .add_modifier(Modifier::BOLD);
                        } else if is_selected_row {
                            style = style.bg(Color::DarkGray);
                        } else if is_selected_col {
                            style = style.bg(Color::DarkGray);
                        }

                        if !is_selected_cell && self.is_search_match(row_idx, col_idx) {
                            style = style.fg(Color::Magenta).add_modifier(Modifier::BOLD);
                        }

                        Cell::from(content.to_string()).style(style)
                    })
                    .collect::<Vec<_>>();
                Row::new(row_cells)
            })
            .collect::<Vec<_>>();

        if self.selection_active && self.selected_row > 0 {
            self.table_state
                .select(Some(self.selected_row.saturating_sub(1)));
        } else {
            self.table_state.select(None);
        }

        if self.selection_active && !self.headers.is_empty() {
            if let Some(visible_index) = visible_columns
                .iter()
                .position(|&idx| idx == self.selected_col)
            {
                self.table_state.select_column(Some(visible_index));
            } else {
                self.table_state.select_column(None);
            }
        } else {
            self.table_state.select_column(None);
        }

        let table = Table::new(rows, widths)
            .header(header_row)
            .block(Block::bordered().title(format!(
                "Table · {} columns × {} rows",
                self.headers.len(),
                self.rows.len()
            )))
            .column_spacing(1)
            .row_highlight_style(Style::default().bg(Color::DarkGray))
            .column_highlight_style(Style::default().bg(Color::DarkGray))
            .cell_highlight_style(
                Style::default()
                    .bg(Color::Cyan)
                    .fg(Color::Black)
                    .add_modifier(Modifier::BOLD),
            );

        frame.render_stateful_widget(table, area, &mut self.table_state);

        if column_widths.len() > visible_columns.len() && area.width > 2 && area.height > 1 {
            let mut scrollbar_state = ScrollbarState::new(column_widths.len())
                .position(self.column_offset)
                .viewport_content_length(visible_columns.len());
            let scrollbar_area = Rect {
                x: area.x + 1,
                y: area.y + area.height.saturating_sub(1),
                width: area.width.saturating_sub(2),
                height: 1,
            };
            if scrollbar_area.width > 0 {
                let scrollbar = Scrollbar::new(ScrollbarOrientation::HorizontalBottom);
                frame.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
            }
        }
    }

    fn render_chart(&self, frame: &mut Frame, area: Rect) {
        if self.headers.is_empty() {
            frame.render_widget(
                Paragraph::new("No data to chart")
                    .alignment(Alignment::Center)
                    .block(Block::bordered().title("Chart")),
                area,
            );
            return;
        }

        match self.chart_mode {
            ChartMode::Line => self.render_line_chart(frame, area),
            ChartMode::Histogram => self.render_histogram(frame, area),
            ChartMode::Pie => self.render_pie_chart(frame, area),
        }
    }

    fn render_line_chart(&self, frame: &mut Frame, area: Rect) {
        let column = match self.headers.get(self.selected_col) {
            Some(name) => name.clone(),
            None => {
                self.render_chart_message(frame, area, "Select a valid column for charts.");
                return;
            }
        };

        let values = self.numeric_values(self.selected_col);
        if values.is_empty() {
            self.render_chart_message(
                frame,
                area,
                "Column has no numeric data to plot as a line chart.",
            );
            return;
        }

        let points = values
            .iter()
            .enumerate()
            .map(|(idx, value)| (idx as f64 + 1.0, *value))
            .collect::<Vec<_>>();

        let min = values
            .iter()
            .copied()
            .fold(f64::INFINITY, |acc, v| acc.min(v));
        let max = values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |acc, v| acc.max(v));
        let (min, max) = expand_range(min, max);

        let dataset = Dataset::default()
            .name(column.clone())
            .data(&points)
            .graph_type(GraphType::Line)
            .marker(symbols::Marker::Dot)
            .style(Style::default().fg(Color::Cyan));

        let len = values.len().max(1);
        let mid = ((len + 1) / 2).max(1);
        let x_labels = [
            Line::from("1"),
            Line::from(format!("{mid}")),
            Line::from(format!("{len}")),
        ];
        let y_mid = (min + max) / 2.0;
        let y_labels = [
            Line::from(format!("{min:.2}")),
            Line::from(format!("{y_mid:.2}")),
            Line::from(format!("{max:.2}")),
        ];

        let chart = Chart::new(vec![dataset])
            .block(Block::bordered().title(format!("Line Chart · {}", column)))
            .x_axis(
                Axis::default()
                    .title("Row #")
                    .bounds([1.0, len as f64])
                    .labels(x_labels),
            )
            .y_axis(
                Axis::default()
                    .title("Value")
                    .bounds([min, max])
                    .labels(y_labels),
            );

        frame.render_widget(chart, area);
    }

    fn render_histogram(&self, frame: &mut Frame, area: Rect) {
        let column = match self.headers.get(self.selected_col) {
            Some(name) => name.clone(),
            None => {
                self.render_chart_message(frame, area, "Select a valid column for charts.");
                return;
            }
        };

        let values = self.numeric_values(self.selected_col);
        if values.len() < 2 {
            self.render_chart_message(
                frame,
                area,
                "Need at least two numeric values for a histogram.",
            );
            return;
        }

        let min = values
            .iter()
            .copied()
            .fold(f64::INFINITY, |acc, v| acc.min(v));
        let max = values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |acc, v| acc.max(v));
        let (min, max) = expand_range(min, max);

        let bucket_count = (values.len() as f64).sqrt().ceil() as usize;
        let bucket_count = bucket_count.clamp(3, 12);
        let bucket_width = (max - min) / bucket_count as f64;
        let mut counts = vec![0u64; bucket_count];

        for value in values {
            let mut bucket = if bucket_width == 0.0 {
                0
            } else {
                ((value - min) / bucket_width).floor() as usize
            };
            if bucket >= bucket_count {
                bucket = bucket_count - 1;
            }
            counts[bucket] += 1;
        }

        let mut bars = Vec::with_capacity(bucket_count);
        let mut max_count = 0;
        for (idx, count) in counts.iter().enumerate() {
            let start = min + bucket_width * idx as f64;
            let end = start + bucket_width;
            let label = if bucket_width == 0.0 {
                format!("{start:.2}")
            } else {
                format!("{start:.2}-{end:.2}")
            };
            max_count = max_count.max(*count);
            bars.push(
                Bar::default()
                    .value(*count)
                    .label(Line::from(label.clone()))
                    .text_value(count.to_string())
                    .style(Style::default().fg(Color::LightBlue)),
            );
        }

        let data = BarGroup::default().bars(&bars);
        let chart = BarChart::default()
            .block(Block::bordered().title(format!("Histogram · {}", column)))
            .bar_width(6)
            .bar_gap(1)
            .bar_style(Style::default().fg(Color::LightBlue))
            .value_style(Style::default().fg(Color::White))
            .label_style(Style::default().fg(Color::Gray))
            .data(data)
            .max(max_count.max(1));

        frame.render_widget(chart, area);
    }

    fn render_pie_chart(&self, frame: &mut Frame, area: Rect) {
        let column = match self.headers.get(self.selected_col) {
            Some(name) => name.clone(),
            None => {
                self.render_chart_message(frame, area, "Select a valid column for charts.");
                return;
            }
        };

        let entries = self.category_counts(self.selected_col);
        if entries.is_empty() {
            self.render_chart_message(
                frame,
                area,
                "Column has no categorical data to build a pie chart.",
            );
            return;
        }

        let block = Block::bordered().title(format!("Pie Chart · {}", column));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.width < 10 || inner.height < 5 {
            return;
        }

        let layout = Layout::horizontal([Constraint::Percentage(68), Constraint::Percentage(32)])
            .split(inner);

        let pie_area = layout[0];
        let legend_area = layout.get(1).copied().unwrap_or(Rect {
            x: pie_area.x,
            y: pie_area.y,
            width: 0,
            height: 0,
        });

        let pie_rect = pie_area;
        let canvas = Canvas::default()
            .marker(symbols::Marker::Dot)
            .x_bounds([-1.2, 1.2])
            .y_bounds([-1.2, 1.2])
            .paint(|ctx| draw_pie(ctx, &entries, pie_rect));
        frame.render_widget(canvas, pie_area);

        if legend_area.height > 0 && legend_area.width > 0 {
            let legend_lines = entries
                .iter()
                .map(|entry| {
                    let percent = (entry.fraction * 100.0).round() as u64;
                    let label = truncate_label(&entry.label, 18);
                    Line::from(vec![
                        Span::styled(
                            "[] ",
                            Style::default()
                                .fg(entry.color)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::raw(format!("{:<18} {:>3}% ({})", label, percent, entry.count)),
                    ])
                })
                .collect::<Vec<_>>();

            frame.render_widget(
                Paragraph::new(legend_lines)
                    .wrap(Wrap { trim: true })
                    .block(Block::default().title("Data")),
                legend_area,
            );
        }
    }

    fn render_chart_message(&self, frame: &mut Frame, area: Rect, message: &str) {
        frame.render_widget(
            Paragraph::new(message)
                .alignment(Alignment::Center)
                .block(Block::bordered().title(format!("{} Chart", self.chart_mode.label()))),
            area,
        );
    }

    fn handle_crossterm_events(&mut self) -> Result<()> {
        match event::read()? {
            Event::Key(key) if key.kind == KeyEventKind::Press => self.on_key_event(key),
            Event::Mouse(_) => {}
            Event::Resize(_, _) => {}
            _ => {}
        }
        Ok(())
    }

    fn on_key_event(&mut self, key: KeyEvent) {
        if self.handle_global_shortcuts(&key) {
            return;
        }

        if matches!(self.input_mode, InputMode::Searching(_)) {
            self.handle_search_key(key);
        } else {
            self.handle_normal_key(key);
        }

        self.ensure_selection_in_bounds();
    }

    fn handle_global_shortcuts(&mut self, key: &KeyEvent) -> bool {
        match (key.modifiers, key.code) {
            (KeyModifiers::CONTROL, KeyCode::Char('c') | KeyCode::Char('C')) => {
                self.quit();
                true
            }
            _ => false,
        }
    }

    fn handle_search_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.input_mode = InputMode::Normal;
                self.set_status("Search canceled");
            }
            KeyCode::Enter => self.submit_search_query(),
            KeyCode::Backspace => {
                if let InputMode::Searching(input) = &mut self.input_mode {
                    input.buffer.pop();
                }
            }
            KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let InputMode::Searching(input) = &mut self.input_mode {
                    input.buffer.push(ch);
                }
            }
            _ => {}
        }
    }

    fn handle_normal_key(&mut self, key: KeyEvent) {
        match (key.modifiers, key.code) {
            (_, KeyCode::Esc) => self.handle_escape(),
            (_, KeyCode::Char('q')) | (_, KeyCode::Char('Q')) => self.quit(),
            (_, KeyCode::Char('/')) => self.start_search(),
            (_, KeyCode::Left) => {
                if self.selected_col > 0 {
                    self.selected_col -= 1;
                }
                self.selection_active = true;
            }
            (_, KeyCode::Right) => {
                if self.selected_col + 1 < self.column_count() {
                    self.selected_col += 1;
                }
                self.selection_active = true;
            }
            (_, KeyCode::Up) => {
                if self.selected_row > 0 {
                    self.selected_row -= 1;
                }
                self.selection_active = true;
            }
            (_, KeyCode::Down) => {
                if self.selected_row < self.row_count() {
                    self.selected_row += 1;
                }
                self.selection_active = true;
            }
            (_, KeyCode::Home) => {
                self.selected_col = 0;
                self.selection_active = true;
            }
            (_, KeyCode::End) => {
                if self.column_count() > 0 {
                    self.selected_col = self.column_count() - 1;
                }
                self.selection_active = true;
            }
            (_, KeyCode::PageUp) => {
                self.selected_row = self.selected_row.saturating_sub(10);
                self.selection_active = true;
            }
            (_, KeyCode::PageDown) => {
                let max_idx = self.row_count().saturating_sub(1);
                self.selected_row = (self.selected_row + 10).min(max_idx);
                self.selection_active = true;
            }
            (_, KeyCode::Enter) => {
                if self.advance_search_match() {
                    return;
                }
                if self.selected_row == 0 {
                    self.sort_by_selected_column();
                }
            }
            (_, KeyCode::Char('1')) => self.set_chart_mode(ChartMode::Line),
            (_, KeyCode::Char('2')) => self.set_chart_mode(ChartMode::Histogram),
            (_, KeyCode::Char('3')) => self.set_chart_mode(ChartMode::Pie),
            (_, KeyCode::Char('r')) | (_, KeyCode::Char('R')) => self.reset_sort(),
            (_, KeyCode::Tab) => {
                let next = self.chart_mode.cycle();
                self.set_chart_mode(next);
            }
            _ => {}
        }
    }

    fn handle_escape(&mut self) {
        if self.selection_active {
            self.selection_active = false;
            self.set_status("Selection cleared. Press Esc again to quit");
        } else {
            self.quit();
        }
    }

    fn start_search(&mut self) {
        let scope = if self.selection_active && self.column_count() > 0 {
            SearchScope::Column(self.selected_col.min(self.column_count() - 1))
        } else {
            SearchScope::Global
        };
        self.input_mode = InputMode::Searching(SearchInput::new(scope));
        self.clear_search_results();
        self.search_cursor_visible = true;
        self.last_cursor_toggle = Instant::now();
        let label = scope.label(&self.headers);
        self.set_status(format!("Search [{}]: enter regex pattern", label));
    }

    fn submit_search_query(&mut self) {
        let (scope, pattern) = match &self.input_mode {
            InputMode::Searching(input) => (input.scope, input.buffer.clone()),
            InputMode::Normal => return,
        };

        if pattern.is_empty() {
            self.set_status("Search pattern cannot be empty");
            return;
        }

        let regex = match Regex::new(&pattern) {
            Ok(regex) => regex,
            Err(error) => {
                self.set_status(format!("Invalid regex: {error}"));
                return;
            }
        };

        self.input_mode = InputMode::Normal;

        let matches = self.collect_matches(scope, &regex);
        if matches.is_empty() {
            self.search_results = None;
            self.set_status(format!("No matches for /{}", pattern));
            return;
        }

        self.search_results = Some(SearchResults::new(scope, pattern.clone(), matches));
        self.focus_search_match(0);

        if let Some(results) = &self.search_results {
            let label = results.scope.label(&self.headers);
            self.set_status(format!(
                "Search [{}]: {} match{}",
                label,
                results.len(),
                if results.len() == 1 { "" } else { "es" }
            ));
        }
    }

    fn clear_search_results(&mut self) {
        self.search_results = None;
    }

    fn collect_matches(&self, scope: SearchScope, regex: &Regex) -> Vec<CellPosition> {
        match scope {
            SearchScope::Global => self
                .rows
                .iter()
                .enumerate()
                .flat_map(|(row_idx, row)| {
                    row.iter().enumerate().filter_map(move |(col_idx, value)| {
                        if regex.is_match(value) {
                            Some(CellPosition {
                                row: row_idx,
                                col: col_idx,
                            })
                        } else {
                            None
                        }
                    })
                })
                .collect(),
            SearchScope::Column(col) => self
                .rows
                .iter()
                .enumerate()
                .filter_map(|(row_idx, row)| {
                    row.get(col).and_then(|value| {
                        if regex.is_match(value) {
                            Some(CellPosition { row: row_idx, col })
                        } else {
                            None
                        }
                    })
                })
                .collect(),
        }
    }

    fn focus_search_match(&mut self, index: usize) {
        if let Some(results) = self.search_results.as_mut() {
            if results.matches.is_empty() {
                return;
            }
            let idx = index % results.matches.len();
            results.current_index = idx;
            if let Some(target) = results.matches.get(idx).copied() {
                self.selection_active = true;
                self.selected_row = target.row.saturating_add(1);
                self.selected_col = target.col;
                self.ensure_selection_in_bounds();
            }
        }
    }

    fn advance_search_match(&mut self) -> bool {
        let next_index = match self.search_results.as_ref() {
            Some(results) if !results.matches.is_empty() => {
                (results.current_index + 1) % results.matches.len()
            }
            _ => return false,
        };

        self.focus_search_match(next_index);
        if let Some(results) = &self.search_results {
            let label = results.scope.label(&self.headers);
            self.set_status(format!(
                "Search [{}]: match {}/{}",
                label,
                results.current_index + 1,
                results.len()
            ));
        }
        true
    }

    fn set_chart_mode(&mut self, mode: ChartMode) {
        if self.chart_mode != mode {
            self.chart_mode = mode;
            self.set_status(format!("Switched to {} chart", mode.label()));
        }
    }

    fn sort_by_selected_column(&mut self) {
        if self.column_count() == 0 {
            return;
        }
        let column = self.selected_col.min(self.column_count() - 1);
        let ascending = if self.sort_column == Some(column) {
            !self.sort_ascending
        } else {
            true
        };

        self.clear_search_results();
        self.rows
            .sort_by(|a, b| compare_cells(a.get(column), b.get(column)));
        if !ascending {
            self.rows.reverse();
        }

        self.sort_column = Some(column);
        self.sort_ascending = ascending;

        let name = self
            .headers
            .get(column)
            .cloned()
            .unwrap_or_else(|| format!("Column {}", column + 1));
        self.set_status(format!(
            "Sorted by {} ({})",
            name,
            if ascending { "asc" } else { "desc" }
        ));
    }

    fn reset_sort(&mut self) {
        if self.headers.is_empty() {
            return;
        }

        if self.sort_column.is_none() || self.rows == self.original_rows {
            self.sort_column = None;
            self.sort_ascending = true;
            self.set_status("Rows already in CSV order");
            return;
        }

        self.clear_search_results();
        self.rows = self.original_rows.clone();
        self.sort_column = None;
        self.sort_ascending = true;
        self.set_status("Sort order reset to CSV order");
    }

    fn ensure_selection_in_bounds(&mut self) {
        let cols = self.column_count();
        if cols == 0 {
            self.selected_col = 0;
            self.column_offset = 0;
        } else if self.selected_col >= cols {
            self.selected_col = cols - 1;
            self.column_offset = self.column_offset.min(cols - 1);
        } else {
            self.column_offset = self.column_offset.min(cols - 1);
        }

        let max_row = self.row_count();
        if max_row == 0 {
            self.selected_row = 0;
        } else if self.selected_row >= max_row {
            self.selected_row = max_row - 1;
        }
    }

    fn row_count(&self) -> usize {
        self.rows.len().saturating_add(1)
    }

    fn column_count(&self) -> usize {
        self.headers.len()
    }

    fn set_status(&mut self, message: impl Into<String>) {
        self.status_message = message.into();
    }

    fn numeric_values(&self, column: usize) -> Vec<f64> {
        self.rows
            .iter()
            .filter_map(|row| {
                row.get(column)
                    .map(|cell| cell.trim())
                    .filter(|cell| !cell.is_empty())
                    .and_then(|cell| cell.parse::<f64>().ok())
            })
            .collect()
    }

    fn category_counts(&self, column: usize) -> Vec<PieEntry> {
        let mut counts: HashMap<String, u64> = HashMap::new();
        for row in &self.rows {
            if let Some(value) = row.get(column) {
                let trimmed = value.trim();
                if !trimmed.is_empty() {
                    *counts.entry(trimmed.to_string()).or_insert(0) += 1;
                }
            }
        }

        if counts.is_empty() {
            return Vec::new();
        }

        let total: u64 = counts.values().sum();
        let mut entries = counts.into_iter().collect::<Vec<_>>();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        const MAX_SEGMENTS: usize = 8;
        let mut result = Vec::new();
        let mut other_count = 0;

        for (idx, (label, count)) in entries.into_iter().enumerate() {
            if idx < MAX_SEGMENTS {
                result.push(PieEntry::new(label, count, total));
            } else {
                other_count += count;
            }
        }

        if other_count > 0 {
            result.push(PieEntry::new("Other".to_string(), other_count, total));
        }

        result
    }

    fn compute_column_widths(&self) -> Vec<u16> {
        let mut widths = Vec::with_capacity(self.headers.len());

        for (idx, header) in self.headers.iter().enumerate() {
            let mut max_width = text_width(header);
            if self.sort_column == Some(idx) {
                max_width = max_width.saturating_add(2);
            }

            for row in &self.rows {
                if let Some(value) = row.get(idx) {
                    let trimmed = value.trim();
                    max_width = max_width.max(text_width(trimmed));
                }
            }

            let padded = max_width.max(3).saturating_add(2);
            widths.push(padded.min(u16::MAX as usize) as u16);
        }

        widths
    }

    fn is_search_match(&self, row_idx: usize, col_idx: usize) -> bool {
        self.search_results
            .as_ref()
            .map(|results| results.is_match(row_idx, col_idx))
            .unwrap_or(false)
    }

    fn adjust_column_offset(&mut self, area_width: u16, column_widths: &[u16]) {
        if column_widths.is_empty() {
            self.column_offset = 0;
            return;
        }

        if self.selected_col >= column_widths.len() {
            self.selected_col = column_widths.len().saturating_sub(1);
        }

        self.column_offset = self
            .column_offset
            .min(column_widths.len().saturating_sub(1));

        let available = area_width.saturating_sub(2);
        if available == 0 {
            self.column_offset = 0;
            return;
        }

        let target = self.selected_col;
        if Self::visible_columns_from_offset(self.column_offset, column_widths, available)
            .contains(&target)
        {
            return;
        }

        for offset in 0..=target {
            if Self::visible_columns_from_offset(offset, column_widths, available).contains(&target)
            {
                self.column_offset = offset;
                return;
            }
        }

        self.column_offset = target;
    }

    fn visible_columns(&self, area_width: u16, column_widths: &[u16]) -> Vec<usize> {
        if column_widths.is_empty() {
            return Vec::new();
        }

        let available = area_width.saturating_sub(2);
        if available == 0 {
            return vec![
                self.column_offset
                    .min(column_widths.len().saturating_sub(1)),
            ];
        }

        Self::visible_columns_from_offset(self.column_offset, column_widths, available)
    }

    fn visible_columns_from_offset(
        offset: usize,
        column_widths: &[u16],
        available_width: u16,
    ) -> Vec<usize> {
        const COLUMN_SPACING: u16 = 1;

        if column_widths.is_empty() {
            return Vec::new();
        }

        let mut consumed = 0u16;
        let mut visible = Vec::new();

        for idx in offset..column_widths.len() {
            let col_width = column_widths[idx];
            if visible.is_empty() && col_width > available_width {
                visible.push(idx);
                break;
            }

            let extra = if visible.is_empty() {
                col_width
            } else {
                col_width.saturating_add(COLUMN_SPACING)
            };

            if !visible.is_empty() && consumed.saturating_add(extra) > available_width {
                break;
            }

            consumed = consumed.saturating_add(extra);
            visible.push(idx);

            if consumed >= available_width {
                break;
            }
        }

        if visible.is_empty() && offset < column_widths.len() {
            visible.push(offset);
        }

        visible
    }

    fn quit(&mut self) {
        self.running = false;
    }

    fn load_csv(path: &Path) -> Result<(Vec<String>, Vec<Vec<String>>)> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)?;
        let mut headers = reader
            .headers()?
            .iter()
            .map(|entry| entry.to_string())
            .collect::<Vec<_>>();

        let mut rows: Vec<Vec<String>> = Vec::new();
        let mut max_columns = headers.len();

        for record in reader.records() {
            let record = record?;
            max_columns = max_columns.max(record.len());
            rows.push(record.iter().map(|entry| entry.to_string()).collect());
        }

        if headers.len() < max_columns {
            for idx in headers.len()..max_columns {
                headers.push(format!("Column {}", idx + 1));
            }
        }

        for row in &mut rows {
            if row.len() < max_columns {
                row.resize(max_columns, String::new());
            }
        }

        Ok((headers, rows))
    }
}

#[derive(Debug, Clone)]
struct PieEntry {
    label: String,
    count: u64,
    fraction: f64,
    color: Color,
}

impl PieEntry {
    fn new(label: String, count: u64, total: u64) -> Self {
        let palette = [
            Color::LightCyan,
            Color::LightMagenta,
            Color::LightGreen,
            Color::LightYellow,
            Color::LightBlue,
            Color::LightRed,
            Color::Cyan,
            Color::Magenta,
            Color::Green,
        ];
        // Assign color based on hash to keep consistent between renders.
        let idx = (label
            .bytes()
            .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32))
            % palette.len() as u32) as usize;
        Self {
            label,
            count,
            fraction: if total == 0 {
                0.0
            } else {
                count as f64 / total as f64
            },
            color: palette[idx],
        }
    }
}

fn draw_pie(ctx: &mut CanvasContext<'_>, entries: &[PieEntry], area: Rect) {
    if entries.is_empty() {
        return;
    }

    const RADIUS: f64 = 1.0;
    const ASPECT: f64 = 0.7;
    const X_MIN: f64 = -1.2;
    const X_MAX: f64 = 1.2;
    const Y_MIN: f64 = -1.2;
    const Y_MAX: f64 = 1.2;

    let mut segments = Vec::new();
    let mut start_angle = 0.0;
    for (idx, entry) in entries.iter().enumerate() {
        if entry.fraction <= f64::EPSILON {
            continue;
        }
        let mut end_angle = start_angle + entry.fraction * TAU;
        if idx == entries.len() - 1 {
            end_angle = TAU;
        }
        segments.push((start_angle, end_angle, entry.color));
        start_angle = end_angle;
    }

    if segments.is_empty() {
        return;
    }

    let width = area.width.max(1) as usize;
    let height = area.height.max(1) as usize;
    let x_span = X_MAX - X_MIN;
    let y_span = Y_MAX - Y_MIN;

    {
        let mut painter = Painter::from(&mut *ctx);
        for row in 0..height {
            let y = Y_MAX - (row as f64 + 0.5) / height as f64 * y_span;
            let scaled_y = y / ASPECT;
            let ellipse_y = (scaled_y / RADIUS).powi(2);
            if ellipse_y > 1.0 {
                continue;
            }

            for col in 0..width {
                let x = X_MIN + (col as f64 + 0.5) / width as f64 * x_span;
                let ellipse_x = (x / RADIUS).powi(2);
                if ellipse_x + ellipse_y > 1.0 + f64::EPSILON {
                    continue;
                }

                let mut angle = scaled_y.atan2(x);
                if angle < 0.0 {
                    angle += TAU;
                }

                let color = color_for_angle(angle, &segments);
                painter.paint(col, row, color);
            }
        }
    }

    ctx.draw(&PieOutline::new(RADIUS, ASPECT, Color::Gray));
}

fn color_for_angle(angle: f64, segments: &[(f64, f64, Color)]) -> Color {
    for (idx, (start, end, color)) in segments.iter().enumerate() {
        if angle < *start {
            continue;
        }

        let is_last = idx + 1 == segments.len();
        if angle < *end || (is_last && angle <= *end + 1e-6) {
            return *color;
        }
    }

    segments
        .last()
        .map(|(_, _, color)| *color)
        .unwrap_or(Color::White)
}

fn truncate_label(label: &str, max_chars: usize) -> String {
    let mut result = String::new();
    for (idx, ch) in label.chars().enumerate() {
        if idx >= max_chars {
            result.push_str("...");
            break;
        }
        result.push(ch);
    }
    result
}

fn expand_range(min: f64, max: f64) -> (f64, f64) {
    if min.is_infinite() || max.is_infinite() {
        return (0.0, 1.0);
    }

    let delta = (max - min).abs();
    if delta < f64::EPSILON {
        let adjustment = if min == 0.0 { 1.0 } else { min.abs() * 0.1 };
        (min - adjustment, max + adjustment)
    } else {
        (min, max)
    }
}

fn text_width(text: &str) -> usize {
    text.chars().count()
}

fn compare_cells(left: Option<&String>, right: Option<&String>) -> Ordering {
    let left = left.map(String::as_str).unwrap_or_default().trim();
    let right = right.map(String::as_str).unwrap_or_default().trim();

    let numeric_cmp = match (left.parse::<f64>(), right.parse::<f64>()) {
        (Ok(a), Ok(b)) => a.partial_cmp(&b),
        _ => None,
    };

    if let Some(order) = numeric_cmp {
        return order;
    }

    left.cmp(right)
}

#[derive(Debug, Clone, Copy)]
struct PieOutline {
    radius: f64,
    aspect: f64,
    color: Color,
}

impl PieOutline {
    fn new(radius: f64, aspect: f64, color: Color) -> Self {
        Self {
            radius,
            aspect,
            color,
        }
    }
}

impl Shape for PieOutline {
    fn draw(&self, painter: &mut Painter) {
        let steps = 200;
        for step in 0..=steps {
            let angle = TAU * step as f64 / steps as f64;
            let x = self.radius * angle.cos();
            let y = self.radius * angle.sin() * self.aspect;
            if let Some((px, py)) = painter.get_point(x, y) {
                painter.paint(px, py, self.color);
            }
        }
    }
}
