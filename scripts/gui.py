from tkinter import (
    Tk,
    Frame,
    Button,
    Label,
    Scrollbar,
    Canvas,
    NW,
    BOTH,
    LEFT,
    RIGHT,
    Y,
    VERTICAL,
    END,
)
from data_processing import load_and_process_data, get_highest_lowest_scores
from model_training import train_model
from plot_functions import (
    plot_feature_importance,
    plot_category_over_time,
    plot_all_categories_over_time,
    plot_correlation_heatmap,
    plot_box_plots,
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Set a consistent background color
BACKGROUND_COLOR = "#f0f0f0"
PLOT_BACKGROUND_COLOR = "#f0f0f0"


def create_gui():
    global root
    root = Tk()
    root.title("DAYMOOD")
    root.geometry("1400x900")
    root.minsize(1400, 900)
    root.resizable(True, True)

    load_main_menu()

    root.mainloop()


def load_main_menu():
    for widget in root.winfo_children():
        widget.destroy()

    main_frame = Frame(root, bg=BACKGROUND_COLOR)
    main_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)

    title = Label(
        main_frame, text="DAYMOOD", font=("Helvetica", 28, "bold"), bg=BACKGROUND_COLOR
    )
    title.pack(pady=20)

    sub_frame = Frame(main_frame, bg=BACKGROUND_COLOR)
    sub_frame.pack(fill=BOTH, expand=True)

    general_frame = Frame(sub_frame, bg=BACKGROUND_COLOR)
    general_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=20, pady=20)

    individual_frame = Frame(sub_frame, bg=BACKGROUND_COLOR)
    individual_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=20, pady=20)

    general_title = Label(
        general_frame,
        text="General Options",
        font=("Helvetica", 16, "bold"),
        bg=BACKGROUND_COLOR,
    )
    general_title.pack(pady=10)

    individual_title = Label(
        individual_frame,
        text="Category Plots",
        font=("Helvetica", 16, "bold"),
        bg=BACKGROUND_COLOR,
    )
    individual_title.pack(pady=10)

    data, data_cleaned = load_and_process_data()
    features_to_exclude = ["Mood", "Day", "Date"]
    feature_importance_df = train_model(data_cleaned, features_to_exclude)
    categories, highest_lowest_df = get_highest_lowest_scores(
        data_cleaned, features_to_exclude
    )

    btn_style = {"font": ("Helvetica", 12), "pady": 10, "bg": BACKGROUND_COLOR}

    btn_highest_lowest = Button(
        general_frame,
        text="Show Highest and Lowest Scores",
        command=lambda: show_highest_lowest(highest_lowest_df),
        **btn_style,
    )
    btn_highest_lowest.pack(pady=10, fill="x")

    btn_plot_importance = Button(
        general_frame,
        text="Plot Feature Importance",
        command=lambda: show_plot(
            lambda: plot_feature_importance(feature_importance_df)
        ),
        **btn_style,
    )
    btn_plot_importance.pack(pady=10, fill="x")

    btn_plot_all_categories = Button(
        general_frame,
        text="Plot All Categories Over Time",
        command=lambda: show_plot(
            lambda: plot_all_categories_over_time(
                data_cleaned, categories, features_to_exclude
            )
        ),
        **btn_style,
    )
    btn_plot_all_categories.pack(pady=10, fill="x")

    btn_plot_correlation_heatmap = Button(
        general_frame,
        text="Plot Correlation Heatmap",
        command=lambda: show_plot(lambda: plot_correlation_heatmap(data_cleaned)),
        **btn_style,
    )
    btn_plot_correlation_heatmap.pack(pady=10, fill="x")

    btn_plot_box_plots = Button(
        general_frame,
        text="Plot Box Plots",
        command=lambda: show_plot(lambda: plot_box_plots(data_cleaned)),
        **btn_style,
    )
    btn_plot_box_plots.pack(pady=10, fill="x")

    for category in categories:
        if category not in features_to_exclude:
            btn = Button(
                individual_frame,
                text=f"Plot {category} Over Time",
                command=lambda c=category: show_plot(
                    lambda: plot_category_over_time(data_cleaned, c)
                ),
                **btn_style,
            )
            btn.pack(pady=5, fill="x")

    footer = Label(
        main_frame,
        text="Created by jaydchw",
        font=("Helvetica", 10, "italic"),
        bg=BACKGROUND_COLOR,
    )
    footer.pack(side="bottom", pady=20)


def show_plot(plot_function):
    for widget in root.winfo_children():
        widget.destroy()

    plot_frame = Frame(root, bg=BACKGROUND_COLOR)
    plot_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)

    back_button = Button(
        plot_frame,
        text="Back",
        command=load_main_menu,
        font=("Helvetica", 12),
        bg=BACKGROUND_COLOR,
    )
    back_button.pack(anchor=NW, padx=10, pady=10)

    fig = plot_function()
    fig.patch.set_facecolor(PLOT_BACKGROUND_COLOR)
    ax = fig.gca()
    ax.set_facecolor(PLOT_BACKGROUND_COLOR)

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=BOTH, expand=True)


def show_highest_lowest(highest_lowest_df):
    for widget in root.winfo_children():
        widget.destroy()

    frame = Frame(root, bg=BACKGROUND_COLOR)
    frame.pack(fill=BOTH, expand=True, padx=20, pady=20)

    back_button = Button(
        frame,
        text="Back",
        command=load_main_menu,
        font=("Helvetica", 12),
        bg=BACKGROUND_COLOR,
    )
    back_button.pack(anchor=NW, padx=10, pady=10)

    canvas = Canvas(frame, bg=BACKGROUND_COLOR)
    scrollbar = Scrollbar(frame, orient=VERTICAL, command=canvas.yview)
    scrollable_frame = Frame(canvas, bg=BACKGROUND_COLOR)

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor=NW)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=LEFT, fill=BOTH, expand=True)
    scrollbar.pack(side=RIGHT, fill=Y)

    for index, row in highest_lowest_df.iterrows():
        category_frame = Frame(scrollable_frame, bg=BACKGROUND_COLOR)
        category_frame.pack(fill=BOTH, expand=True, pady=5)

        category_label = Label(
            category_frame,
            text=index,
            font=("Helvetica", 14, "bold"),
            bg=BACKGROUND_COLOR,
        )
        category_label.pack(anchor=NW)

        highest_label = Label(
            category_frame,
            text=f"Highest: {row['Highest Day']}",
            font=("Helvetica", 12),
            bg=BACKGROUND_COLOR,
        )
        highest_label.pack(anchor=NW)

        lowest_label = Label(
            category_frame,
            text=f"Lowest: {row['Lowest Day']}",
            font=("Helvetica", 12),
            bg=BACKGROUND_COLOR,
        )
        lowest_label.pack(anchor=NW)


if __name__ == "__main__":
    create_gui()
