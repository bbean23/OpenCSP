import copy
import datetime as dt
import re
from typing import Iterator

import matplotlib.pyplot
import numpy as np
import numpy.typing as npt
import pandas
import pandas.core

from opencsp.common.lib.opencsp_path import opencsp_settings
import opencsp.common.lib.render.color as color
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render.View3d as v3d
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcf
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.time_date_tools as tdt
import opencsp.common.lib.tool.typing_tools as tt


class HeliostatLogsParser:
    def __init__(
        self, name: str, dtype: dict[str, any], date_column_formats: dict[str, str], heliostat_name_column: str
    ):
        # register inputs
        self.name = name
        self.dtype = dtype
        self.date_column_formats = date_column_formats
        self.heliostat_name_column = heliostat_name_column

        # internal values
        self.filename_datetime_replacement: tuple[re.Pattern, str] = None
        self.filename_datetime_format: str = None

        # plotting values
        self.figure_rec: rcfr.RenderControlFigureRecord = None
        self.nplots = 0
        self.parent_parser: HeliostatLogsParser = None

    @classmethod
    def NsttfLogsParser(cls):
        dtype = {
            # "Main T": ,
            "Time": str,
            "Helio": str,
            "Mode": str,
            "Sleep": str,
            "Track": int,
            "X Targ": float,
            "Y Targ": float,
            "Z Targ": float,
            "az offset": float,
            "el offset": float,
            # "reserved": ,
            "Az Targ": float,
            "El Targ": float,
            "Az": float,
            "Elev": float,
            # "Az Amp": ,
            # "El Amp": ,
            # "Az Falt": ,
            # "El Falt": ,
            # "Az Cnt": ,
            # "El Cnt": ,
            # "Az Drive": ,
            # "El Drive": ,
            "Trk Time": float,
            "Exec Time": float,
            "Delta Time": float,
            # "Ephem Num": ,
            # "Status Word": ,
        }
        # date format for "09:59:59.999" style timestamp
        # https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
        date_column_formats = {"Time ": r"%H:%M:%S.%f"}
        heliostat_name_column = "Helio"

        return cls('NsttfLogsParser', dtype, date_column_formats, heliostat_name_column)

    @property
    def column_names(self) -> list[str]:
        return list(self.data.columns)

    @property
    def heliostats(self) -> tt.Series[str]:
        return self.column(self.heliostat_name_column)

    @property
    def datetimes(self) -> tt.Series[dt.datetime]:
        dt_col = next(iter(self.date_column_formats.keys()))
        return self.column(dt_col)

    @datetimes.setter
    def datetimes(self, datetimes: pandas.Series):
        dt_col = next(iter(self.date_column_formats.keys()))
        self.data[dt_col] = datetimes

    def column(self, column_name: str) -> pandas.Series:
        if column_name is None:
            lt.error_and_raise(ValueError, "Error in HeliostatLogsParser.column(): column_name is None")
        if column_name not in self.column_names:
            lt.error_and_raise(
                KeyError,
                "Error in HeliostatLogsParser.column(): "
                + f"can't find column \"{column_name}\", should be one of {self.column_names}",
            )
        return self.data[column_name]

    def load_heliostat_logs(self, log_path_name_exts: str | list[str], usecols: list[str] = None, nrows: int = None):
        # normalize input
        if isinstance(log_path_name_exts, str):
            log_path_name_exts = [log_path_name_exts]
        for i, log_path_name_ext in enumerate(log_path_name_exts):
            log_path_name_exts[i] = ft.norm_path(log_path_name_ext)

        # validate input
        for log_path_name_ext in log_path_name_exts:
            if not ft.file_exists(log_path_name_ext):
                lt.error_and_raise(
                    FileNotFoundError,
                    f"Error in HeliostatLogsParser({self.name}).load_heliostat_logs(): "
                    + f"file \"{log_path_name_ext}\" does not exist!",
                )

        # load the logs
        data_list: list[pandas.DataFrame] = []
        for i, log_path_name_ext in enumerate(log_path_name_exts):
            lt.info(f"Loading {log_path_name_ext}... ")

            data = pandas.read_csv(
                log_path_name_ext,
                delimiter="\t",
                header='infer',
                # parse_dates=self.parse_dates,
                dtype=self.dtype,
                skipinitialspace=True,
                # date_format=self.date_format,
                usecols=usecols,
                nrows=nrows,
            )
            data_list.append(data)
        self.data = pandas.concat(data_list)
        data_list.clear()

        # try to guess the date from the file name
        date = None
        if self.filename_datetime_format is not None:
            _, log_name, _ = ft.path_components(log_path_name_ext)
            if self.filename_datetime_replacement is not None:
                repl_pattern, repl_sub = self.filename_datetime_replacement
                formatted_log_name: str = repl_pattern.sub(repl_sub, log_name)
                date = formatted_log_name
            else:
                date = log_name

        # parse any necessary dates
        # masterlog _ 5_ 3_2024_13.lvm
        for date_col in self.date_column_formats:
            dt_format = self.date_column_formats[date_col]
            col_to_parse = self.data[date_col]
            if not r"%d" in dt_format and r"%j" not in dt_format:
                if date is not None:
                    col_to_parse = date + " " + self.data[date_col]
                    dt_format = self.filename_datetime_format + " " + dt_format

            self.data[date_col] = pandas.to_datetime(col_to_parse, format=dt_format)

        lt.info("..done")

    def filter(
        self,
        heliostat_names: str | list[str] = None,
        columns_equal: list[tuple[str, any]] = None,
        columns_almost_equal: list[tuple[str, float]] = None,
        datetime_range: tuple[dt.datetime, dt.datetime] | tuple[dt.time, dt.time] = None,
    ) -> "HeliostatLogsParser":
        if isinstance(heliostat_names, str):
            heliostat_names = [heliostat_names]

        # copy of the data to be filtered
        new_data = self.data
        if heliostat_names is not None:
            new_data = new_data[new_data[self.heliostat_name_column].isin(heliostat_names)]

        # filter by datetime
        if datetime_range is not None:
            dt_col = next(iter(self.date_column_formats.keys()))
            if isinstance(datetime_range[0], dt.datetime):
                # user specified dates+times
                matches = (new_data[dt_col] >= datetime_range[0]) & (new_data[dt_col] < datetime_range[1])
            elif isinstance(datetime_range[0], dt.time):
                # user specified just times, select by all matches across all dates
                dates: set[dt.date] = set([val.date() for val in self.datetimes])
                matches = np.full_like(new_data[dt_col], fill_value=False, dtype=np.bool_)
                for date in dates:
                    fromval = pandas.to_datetime(dt.datetime.combine(date, datetime_range[0]))
                    toval = pandas.to_datetime(dt.datetime.combine(date, datetime_range[1]))
                    matches |= (new_data[dt_col] >= fromval) & (new_data[dt_col] < toval)
            else:
                lt.error_and_raise(
                    ValueError,
                    "Error in HeliostatLogsParser.filter(): "
                    + f"unexpected type for datetime_range, expected datetime or time but got {type(datetime_range[0])}",
                )
            new_data = new_data[matches]

        # filter by generic exact values
        if columns_equal is not None:
            for column_name, value in columns_almost_equal:
                new_data = new_data[new_data[column_name] == value]

        # filter by generic approximate values
        if columns_almost_equal is not None:
            # definition for 'almost_equal' from np.testing.assert_almost_equal()
            # abs(desired-actual) < float64(1.5 * 10**(-decimal))
            decimal = 7
            error_bar = 1.5 * 10 ** (-decimal)
            for column_name, value in columns_almost_equal:
                matches = np.abs(new_data[column_name] - value) < error_bar
                new_data = new_data[matches]

        # create a copy with the filtered data
        ret = copy.copy(self)
        ret.data = new_data
        ret.parent_parser = self

        return ret

    def check_for_missing_heliostats(self, expected_heliostat_names: list[str]) -> tuple[list[str], list[str]]:
        extra_hnames, missing_hnames = [], copy.copy(expected_heliostat_names)
        hnames = set(self.data["Helio"])
        for hname in hnames:
            if hname in missing_hnames:
                missing_hnames.remove(hname)
            else:
                extra_hnames.append(hname)

        lt.info(f"Missing {len(missing_hnames)} expected heliostats: {missing_hnames}")
        lt.info(f"Found {len(extra_hnames)} extra heliostats: {extra_hnames}")

        return missing_hnames, extra_hnames

    def prepare_figure(self, title: str = None, x_label: str = None, y_label: str = None):
        # normalize input
        if title is None:
            title = f"{self.__class__.__name__} ({self.name})"
        if x_label is None:
            x_label = "x"
        if y_label is None:
            y_label = "y"

        # get the plot ready
        view_spec = vs.view_spec_pq()
        axis_control = rca.RenderControlAxis(x_label=x_label, y_label=y_label)
        figure_control = rcf.RenderControlFigure(tile=False)
        self.fig_record = fm.setup_figure(
            figure_control=figure_control,
            axis_control=axis_control,
            view_spec=view_spec,
            equal=False,
            number_in_name=False,
            title=title,
            code_tag=f"{__file__}.build_plot()",
        )
        self.nplots = 0

        return self.fig_record

    def plot(self, x_axis_column: str, series_columns_labels: dict[str, str], scatter_plot=False):
        view = self.fig_record.view
        x_values = self.data[x_axis_column].to_list()

        # populate the plot
        for series_column in series_columns_labels:
            series_label = series_columns_labels[series_column]
            series_values = self.data[series_column].to_list()
            if scatter_plot:
                view.draw_pq(
                    (x_values, series_values),
                    label=series_label,
                    style=rcps.default(color=color._PlotColors()[self.nplots]),
                )
            else:
                view.draw_pq_list(
                    list(zip(x_values, series_values)),
                    label=series_label,
                    style=rcps.outline(color=color._PlotColors()[self.nplots]),
                )
            self.nplots += 1

        # bubble up the nplots value to the parent parser
        curr_parser = self
        while curr_parser.parent_parser is not None:
            curr_parser.parent_parser.nplots = curr_parser.nplots
            curr_parser = curr_parser.parent_parser


if __name__ == "__main__":
    experiment_path = ft.join(
        opencsp_settings["opencsp_root_path"]["collaborative_dir"],
        "NSTTF_Optics/Experiments/2024-06-16_FluxMeasurement",
    )
    log_path = ft.join(experiment_path, "2_Data/context/heliostat_logs")
    log_name_exts = ft.files_in_directory(log_path)
    log_path_name_exts = [ft.join(log_path, log_name_ext) for log_name_ext in log_name_exts]
    # example log name: "log_ 5_ 3_2024_13" for May 3rd 2024 at 1pm
    # replacement: "2024/5/3"
    save_path = ft.join(experiment_path, "4_Analysis/maybe_slow_13_more_time")
    from_regex = re.compile(r".*_ ?([0-9]{1,2})_ ?([0-9]{1,2})_([0-9]{4})_ ?([0-9]{1,2})")
    to_pattern = r"\3/\1/\2"

    rows = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    ncols = [9, 9, 9, 10, 11, 12, 14, 14, 14, 6]
    expected_hnames = []
    for row, ncols in zip(rows, ncols):
        for col in range(1, ncols + 1):
            for ew in ["E", "W"]:
                expected_hnames.append(f"_{row:02d}{ew}{col:02d}")

    times: dict[tuple[dt.datetime, dt.datetime]] = {
        "_05W01": (dt.time(13, 18, 34), dt.time(13, 18, 54)),
        "_05E06": (dt.time(13, 19, 31), dt.time(13, 19, 52)),
        "_09W01": (dt.time(13, 20, 29), dt.time(13, 20, 49)),
        "_14W01": (dt.time(13, 21, 21), dt.time(13, 21, 44)),
        "_14W06": (dt.time(13, 22, 31), dt.time(13, 22, 56)),
    }
    to_datetime = lambda time: dt.datetime.combine(dt.date(2024, 6, 16), time)
    for hel in times:
        enter, leave = times[hel]
        times[hel] = to_datetime(enter), to_datetime(leave)
    total_delta = times["_14W06"][1] - times["_05W01"][0]
    lt.info(f"{total_delta=}")

    parser = HeliostatLogsParser.NsttfLogsParser()
    parser.filename_datetime_replacement = (from_regex, to_pattern)
    parser.filename_datetime_format = r"%Y/%m/%d"
    parser.load_heliostat_logs(
        log_path_name_exts, usecols=["Helio", "Time ", "X Targ", "Z Targ", "Az", "Az Targ", "Elev", "El Targ"]
    )
    parser.check_for_missing_heliostats(expected_hnames)

    maybe_slow_helios = [
        "_06W03",
        "_6E08",
        "_08E08",
        "_08W08",
        "_10E10",
        "_10W08",
        "_11E11",
        "_11E07",
        "_11W09",
        "_12W05",
        "_12E10",
        "_13E09",
        "_14E06",
    ]
    series_columns_labels_list = [
        {'Az': "{helio}Az", 'Az Targ': "{helio}AzTarg"},
        {'Elev': "{helio}El", 'El Targ': "{helio}ElTarg"},
    ]

    for heliostat in maybe_slow_helios:
        helio = heliostat.lstrip("_")

        for series_columns_labels in series_columns_labels_list[1:]:
            title = helio + " " + ",".join([s for s in series_columns_labels])
            for series in series_columns_labels:
                series_columns_labels[series] = series_columns_labels[series].replace("{helio}", helio)

            fig_record = parser.prepare_figure(title, "Time", "X Targ (m)")
            hparser = parser.filter(
                heliostat_names=heliostat, datetime_range=(dt.time(13, 26, 50), dt.time(13, 29, 30))
            )
            hparser.plot('Time ', series_columns_labels, scatter_plot=False)

            # # datetimes = hparser.datetimes
            # # if len(datetimes) > 4:
            # #     xticks = []
            # #     for i in range(0, len(datetimes), int(len(datetimes) / 4)):
            # #         datetime: pandas.DatetimeIndex = datetimes[i]
            # #         xticks.append((datetime, f"{datetime.time}"))
            # #     fig_record.view.axis.set_xticks([tick_label[0] for tick_label in xticks], [
            # #                                     tick_label[1] for tick_label in xticks])
            # xticks = [dt.datetime(2024, 6, 16, 13, 26, 50) + dt.timedelta(s) for s in range(0, 80, 20)]
            # xlabels = [str(xtick) for xtick in xticks]
            # fig_record.view.axis.set_xticks(xticks, xlabels)

            fig_record.view.show(legend=True, block=False, grid=True)
            fig_record.save(save_path, title, "png")
