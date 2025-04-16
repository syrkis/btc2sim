#import "@preview/touying:0.6.1": *
#import "@local/lilka:0.0.0": *
#import "@preview/simplebnf:0.1.1": *
#show: lilka

#let title = "btc2sim"
#let info = (author: "Noah Syrkis", date: datetime.today(), title: title)
#show: slides.with(config-info(..info), config-common(handout: false))
#metadata((title: title, slug: "miiii"))

#title-slide()



= Grammar

#figure(
  bnf(
    Prod(
      delim: $→$,

      emph[tree],
      {
        Or[#emph[node] $(triangle.stroked.r$ #emph[node])$star$][program]
      },
    ),
    Prod(
      delim: $→$,
      emph[node],
      {
        Or[#emph[S] | #emph[F] | #emph[A] | #emph[C]][node or leaf]
      },
    ),
    Prod(
      delim: $→$,
      emph[S],
      {
        Or[S (#emph[tree])][sequence operator]
      },
    ),
    Prod(
      delim: $→$,
      emph[F],
      {
        Or[F (#emph[tree])][fallback operator]
      },
    ),
    Prod(
      delim: $→$,
      emph[A],
      {
        Or[A (#emph[move] | #emph[stand])][action operator]
      },
    ),
    Prod(
      delim: $→$,
      emph[C],
      {
        Or[C (is_alive | in_range)][condition operator]
      },
    ),
    Prod(
      delim: $→$,
      emph[move],
      {
        Or[#emph[move direction]][move action]
      },
    ),
    Prod(
      delim: $→$,
      emph[direction],
      {
        Or[to | from][direction]
      },
    ),
  ),
  caption: [The BTC2SIM DSL grammar],
)


#page[
  #figure(
    ```
    F (
      A move to king |>
      S ( C is_alive |> A move from queen ) |>
      A move to pawn
    )
    ```,
    caption: [Example program],
  )
]
