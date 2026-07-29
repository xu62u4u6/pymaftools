from pymaftools.ai_install import ASSET_ROOT, install, main


def test_bundled_assets_exist():
    assert (ASSET_ROOT / "agents" / "bioinformatics-researcher.toml").is_file()
    assert (ASSET_ROOT / "skills" / "pymaftools" / "SKILL.md").is_file()


def test_codex_project_install(tmp_path):
    actions = install(
        target="codex",
        scope="project",
        components={"agents", "skill"},
        project_dir=tmp_path,
        dry_run=False,
        force=False,
    )

    assert actions
    assert (tmp_path / ".codex" / "agents" / "bioinformatics-researcher.toml").is_file()
    assert (tmp_path / ".codex" / "skills" / "pymaftools" / "SKILL.md").is_file()


def test_claude_project_install_renders_markdown(tmp_path):
    install(
        target="claude",
        scope="project",
        components={"agents"},
        project_dir=tmp_path,
        dry_run=False,
        force=False,
    )

    content = (
        tmp_path / ".claude" / "agents" / "bioinformatics-researcher.md"
    ).read_text(encoding="utf-8")
    assert content.startswith("---\nname: bioinformatics-researcher\n")
    assert "Do not certify your own conclusions." in content


def test_dry_run_does_not_write(tmp_path):
    actions = install(
        target="codex",
        scope="project",
        components={"agents"},
        project_dir=tmp_path,
        dry_run=True,
        force=False,
    )

    assert actions[0].startswith("WOULD WRITE")
    assert not (tmp_path / ".codex").exists()


def test_install_refuses_to_overwrite_different_file(tmp_path):
    destination = tmp_path / ".codex" / "agents" / "bioinformatics-researcher.toml"
    destination.parent.mkdir(parents=True)
    destination.write_text("user-owned content\n", encoding="utf-8")

    try:
        install(
            target="codex",
            scope="project",
            components={"agents"},
            project_dir=tmp_path,
            dry_run=False,
            force=False,
        )
    except FileExistsError as error:
        assert str(destination) in str(error)
    else:
        raise AssertionError("install should protect a different existing file")

    assert destination.read_text(encoding="utf-8") == "user-owned content\n"


def test_cli_lists_assets(capsys):
    assert main(["list"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "agent bioinformatics-researcher",
        "skill pymaftools",
    ]
