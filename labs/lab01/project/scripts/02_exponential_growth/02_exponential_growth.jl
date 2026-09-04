using DrWatson
@quickactivate "project"

using DifferentialEquations
using DataFrames
using Plots
using JLD2
using BenchmarkTools

script_name = splitext(basename(PROGRAM_FILE))[1]

mkpath(plotsdir(script_name))
mkpath(datadir(script_name))

function exponential_growth!(du, u, p, t)
    du[1] = p.α * u[1]
end

base_params = Dict(
    :u0 => [1.0],
    :α => 0.3,
    :tspan => (0.0, 10.0),
    :solver => Tsit5(),
    :saveat => 0.1,
    :experiment_name => "base_experiment"
)

println("Базовые параметры:")
for (key, value) in base_params
    println("  $key = $value")
end

function run_single_experiment(params::Dict)
    @unpack u0, α, tspan, solver, saveat = params

    problem = ODEProblem(
        exponential_growth!,
        u0,
        tspan,
        (α=α,)
    )

    solution = solve(problem, solver; saveat=saveat)

    final_population = last(solution.u)[1]
    doubling_time = log(2) / α

    return Dict(
        "solution" => solution,
        "time_points" => solution.t,
        "population_values" => first.(solution.u),
        "final_population" => final_population,
        "doubling_time" => doubling_time,
        "parameters" => params
    )
end

base_data, base_path = produce_or_load(
    datadir(script_name, "single"),
    base_params,
    run_single_experiment;
    prefix="exp_growth",
    tag=false,
    verbose=true
)

println("\nРезультаты базового эксперимента:")
println("Финальная популяция: ", base_data["final_population"])
println(
    "Время удвоения: ",
    round(base_data["doubling_time"]; digits=2)
)
println("Файл результатов: ", base_path)

p1 = plot(
    base_data["time_points"],
    base_data["population_values"],
    label="α = $(base_params[:α])",
    xlabel="Время, t",
    ylabel="Популяция, u(t)",
    title="Экспоненциальный рост",
    lw=2,
    legend=:topleft,
    grid=true
)

savefig(plotsdir(script_name, "single_experiment.png"))

param_grid = Dict(
    :u0 => [[1.0]],
    :α => [0.1, 0.3, 0.5, 0.8, 1.0],
    :tspan => [(0.0, 10.0)],
    :solver => [Tsit5()],
    :saveat => [0.1],
    :experiment_name => ["parametric_scan"]
)

all_params = dict_list(param_grid)

println("\n", "="^60)
println("ПАРАМЕТРИЧЕСКОЕ СКАНИРОВАНИЕ")
println("="^60)
println("Количество экспериментов: ", length(all_params))
println("Значения α: ", param_grid[:α])

all_results = []
all_dfs = []

for (i, params) in enumerate(all_params)

    println(
        "Прогресс: $i/$(length(all_params)), α = $(params[:α])"
    )

    data, path = produce_or_load(
        datadir(script_name, "parametric_scan"),
        params,
        run_single_experiment;
        prefix="scan",
        tag=false,
        verbose=false
    )

    summary = merge(
        params,
        Dict(
            :final_population => data["final_population"],
            :doubling_time => data["doubling_time"],
            :filepath => path
        )
    )

    push!(all_results, summary)

    df = DataFrame(
        t=data["time_points"],
        u=data["population_values"],
        α=fill(params[:α], length(data["time_points"]))
    )

    push!(all_dfs, df)
end

results_df = DataFrame(all_results)

println("\nСводная таблица:")
println(
    results_df[
        !,
        [:α, :final_population, :doubling_time]
    ]
)

p2 = plot(
    size=(800, 500),
    dpi=150
)

for params in all_params

    data, _ = produce_or_load(
        datadir(script_name, "parametric_scan"),
        params,
        run_single_experiment;
        prefix="scan"
    )

    plot!(
        p2,
        data["time_points"],
        data["population_values"],
        label="α = $(params[:α])",
        lw=2
    )
end

plot!(
    p2,
    xlabel="Время, t",
    ylabel="Популяция, u(t)",
    title="Влияние α на экспоненциальный рост",
    legend=:topleft,
    grid=true
)

savefig(
    plotsdir(
        script_name,
        "parametric_scan_comparison.png"
    )
)

p3 = plot(
    results_df.α,
    results_df.doubling_time,
    seriestype=:scatter,
    label="Численное решение",
    xlabel="Скорость роста, α",
    ylabel="Время удвоения",
    title="Зависимость времени удвоения от α",
    markersize=8
)

α_range = 0.1:0.01:1.0

plot!(
    p3,
    α_range,
    log(2) ./ α_range,
    label="Теория: ln(2)/α",
    lw=2,
    linestyle=:dash
)

savefig(
    plotsdir(
        script_name,
        "doubling_time_vs_alpha.png"
    )
)

println("\n", "="^60)
println("БЕНЧМАРКИНГ")
println("="^60)

benchmark_results = []

for α_value in param_grid[:α]

    benchmark_params = Dict(
        :u0 => [1.0],
        :α => α_value,
        :tspan => (0.0, 10.0),
        :solver => Tsit5(),
        :saveat => 0.1
    )

    function benchmark_run()
        problem = ODEProblem(
            exponential_growth!,
            benchmark_params[:u0],
            benchmark_params[:tspan],
            (α=benchmark_params[:α],)
        )

        solve(
            problem,
            benchmark_params[:solver];
            saveat=benchmark_params[:saveat]
        )
    end

    benchmark = @benchmark $benchmark_run() samples=100 evals=1

    time_seconds = median(benchmark).time / 1e9

    push!(
        benchmark_results,
        (
            α=α_value,
            time=time_seconds
        )
    )

    println(
        "α = $α_value: ",
        round(time_seconds; digits=4),
        " сек."
    )
end

bench_df = DataFrame(benchmark_results)

p4 = plot(
    bench_df.α,
    bench_df.time,
    seriestype=:scatter,
    label="Время вычисления",
    xlabel="Скорость роста, α",
    ylabel="Время вычисления, сек",
    title="Время вычисления в зависимости от α",
    markersize=8
)

savefig(
    plotsdir(
        script_name,
        "computation_time_vs_alpha.png"
    )
)

@save datadir(script_name, "all_results.jld2") base_params param_grid all_params results_df bench_df

@save datadir(script_name, "all_plots.jld2") p1 p2 p3 p4

println("\n", "="^60)
println("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА")
println("="^60)

println("\nРезультаты:")
println("data/$(script_name)/single/")
println("data/$(script_name)/parametric_scan/")
println("data/$(script_name)/all_results.jld2")
println("plots/$(script_name)/")
println("data/$(script_name)/all_plots.jld2")
